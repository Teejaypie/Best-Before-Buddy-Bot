import os
import io
import json
import base64
import logging
import sqlite3
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
API_KEY = os.getenv("APIYI_API_KEY", "")
BASE_URL = os.getenv(
    "APIYI_BASE_URL", "https://api.apiyi.com/v1"
)
MODEL = os.getenv("APIYI_MODEL", "gpt-4o-mini")
TIMEZONE = os.getenv("TIMEZONE", "Asia/Singapore")
REMINDER_HOUR = int(os.getenv("REMINDER_HOUR", "9"))
DB_PATH = os.getenv("DATABASE_PATH", "food.db")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

if not BOT_TOKEN or not API_KEY:
    raise RuntimeError(
        "Set TELEGRAM_BOT_TOKEN and APIYI_API_KEY in .env"
    )

try:
    TZ = ZoneInfo(TIMEZONE)
except Exception as exc:
    raise RuntimeError(
        "TIMEZONE is invalid. Use an IANA time-zone name."
    ) from exc

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=90.0,
    max_retries=2,
)

# --------------------------------------------------
# 2. DATABASE
# --------------------------------------------------

def db():
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with db() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS foods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                quantity TEXT,
                category TEXT,
                purchase_date TEXT NOT NULL,
                expiry_date TEXT,
                date_source TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL
            )
        """)
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_food_user_status
            ON foods(user_id, status, expiry_date)
        """)


def add_food(
    user_id, name, quantity, category,
    purchase_date, expiry_date, date_source
):
    with db() as con:
        cur = con.execute("""
            INSERT INTO foods (
                user_id, name, quantity, category,
                purchase_date, expiry_date,
                date_source, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)
        """, (
            user_id,
            name,
            quantity,
            category,
            purchase_date.isoformat(),
            expiry_date.isoformat() if expiry_date else None,
            date_source,
            datetime.now(TZ).isoformat(),
        ))
        return cur.lastrowid


def get_food(user_id, food_id):
    with db() as con:
        return con.execute("""
            SELECT * FROM foods
            WHERE id = ? AND user_id = ?
        """, (food_id, user_id)).fetchone()


def list_foods(user_id):
    with db() as con:
        return con.execute("""
            SELECT * FROM foods
            WHERE user_id = ? AND status = 'active'
            ORDER BY
                CASE WHEN expiry_date IS NULL THEN 1 ELSE 0 END,
                expiry_date ASC, name ASC
        """, (user_id,)).fetchall()


def mark_done(user_id, food_id):
    with db() as con:
        cur = con.execute("""
            UPDATE foods SET status = 'consumed'
            WHERE id = ? AND user_id = ? AND status = 'active'
        """, (food_id, user_id))
        return cur.rowcount > 0


def delete_food(user_id, food_id):
    with db() as con:
        cur = con.execute("""
            DELETE FROM foods WHERE id = ? AND user_id = ?
        """, (food_id, user_id))
        return cur.rowcount > 0


def change_expiry(user_id, food_id, new_date):
    with db() as con:
        cur = con.execute("""
            UPDATE foods
            SET expiry_date = ?, date_source = 'user_corrected'
            WHERE id = ? AND user_id = ? AND status = 'active'
        """, (new_date.isoformat(), food_id, user_id))
        return cur.rowcount > 0


# --------------------------------------------------
# 3. IMAGE PROCESSING
# --------------------------------------------------

def image_to_data_url(image_bytes):
    """Convert a Telegram photo into a smaller JPEG data URL."""
    with Image.open(io.BytesIO(image_bytes)) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((1800, 1800))

        output = io.BytesIO()
        image.save(output, format="JPEG", quality=88)

    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


# --------------------------------------------------
# 4. AI RECEIPT READING
# --------------------------------------------------

RECEIPT_PROMPT = """
You read grocery receipts. Extract grocery and food products only.

Return a JSON object with this exact structure:
{
  "receipt_date": "YYYY-MM-DD or null",
  "items": [
    {
      "name": "clear product name",
      "quantity": "quantity as printed, or unknown",
      "category": "produce|dairy|meat|seafood|eggs|bakery|prepared|frozen|pantry|other",
      "printed_expiry_date": "YYYY-MM-DD or null",
      "estimated_days": 7,
      "confidence": "high|medium|low"
    }
  ]
}

Rules:
- Never invent a printed expiry date.
- Receipt purchase dates are not expiry dates.
- If the date is absent or ambiguous, return null.
- estimated_days is only a rough shelf-life suggestion, not a
  verified food-safety limit. Use a conservative estimate where
  appropriate, or null if you cannot make a reasonable estimate.
- Use JSON only. Do not include Markdown fences or explanations.
- Ignore non-food products, payment details, loyalty numbers,
  addresses, and other personal information.
- If the image is not a grocery receipt, return {"receipt_date": null,
  "items": []}.
"""


def parse_json_response(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    return json.loads(text)


def read_receipt(image_bytes):
    data_url = image_to_data_url(image_bytes)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": RECEIPT_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Read this grocery receipt.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ],
        temperature=0,
        max_tokens=2500,
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("The AI returned an empty response.")

    return parse_json_response(content)


# --------------------------------------------------
# 5. DATE VALIDATION AND FOOD-SAFETY GUARDRAILS
# --------------------------------------------------

ALLOWED_CATEGORIES = {
    "produce", "dairy", "meat", "seafood", "eggs",
    "bakery", "prepared", "frozen", "pantry", "other",
}

# Rough default estimates in days, for an ordinary home.
# These are inventory reminders, NOT food-safety guarantees.
DEFAULT_DAYS = {
    "produce": 5,
    "dairy": 5,
    "meat": 2,
    "seafood": 1,
    "eggs": 14,
    "bakery": 3,
    "prepared": 2,
    "frozen": 60,
    "pantry": 30,
    "other": 7,
}


def valid_date(value):
    if not value or not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def clean_text(value, limit=120):
    if not isinstance(value, str):
        return "Unknown"
    return value.strip()[:limit] or "Unknown"


def normalize_items(result, today):
    """Validate model output before writing anything to the database."""
    if not isinstance(result, dict):
        raise ValueError("AI output must be a JSON object.")

    raw_items = result.get("items", [])
    if not isinstance(raw_items, list):
        raise ValueError("AI items must be a list.")

    receipt_date = valid_date(result.get("receipt_date"))
    if not receipt_date or receipt_date > today:
        receipt_date = today

    normalized = []

    for item in raw_items[:80]:
        if not isinstance(item, dict):
            continue

        name = clean_text(item.get("name"), 120)
        if name == "Unknown":
            continue

        category = str(item.get("category", "other")).lower()
        if category not in ALLOWED_CATEGORIES:
            category = "other"

        printed_date = valid_date(item.get("printed_expiry_date"))

        # Reject implausible printed dates; ask the user to correct
        # these rather than trusting a possibly misread year.
        if printed_date and (
            printed_date < receipt_date - timedelta(days=365)
            or printed_date > receipt_date + timedelta(days=3650)
        ):
            printed_date = None

        days = item.get("estimated_days")
        if not isinstance(days, int) or isinstance(days, bool):
            days = DEFAULT_DAYS[category]

        # Prevent extreme or malformed estimates.
        days = max(0, min(days, 365))

        if printed_date:
            expiry_date = printed_date
            source = "printed_on_receipt"
        elif category == "meat" or category == "seafood":
            # Very short provisional reminder only. It is not a
            # substitute for the package date or food-safety rules.
            expiry_date = receipt_date + timedelta(
                days=min(days, DEFAULT_DAYS[category])
            )
            source = "rough_estimate"
        else:
            expiry_date = receipt_date + timedelta(days=days)
            source = "rough_estimate"

        normalized.append({
            "name": name,
            "quantity": clean_text(item.get("quantity"), 80),
            "category": category,
            "purchase_date": receipt_date,
            "expiry_date": expiry_date,
            "date_source": source,
        })

    return normalized, receipt_date


# --------------------------------------------------
# 6. TELEGRAM COMMANDS
# --------------------------------------------------

def is_private(update):
    return (
        update.effective_chat is not None
        and update.effective_chat.type == "private"
        and update.effective_user is not None
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_private(update):
        return

    await update.message.reply_text(
        "Welcome to FreshTrack!\n\n"
        "Send a clear grocery receipt photo to add food.\n\n"
        "Commands:\n"
        "/list - View your food inventory\n"
        "/soon - View food expiring within 3 days\n"
        "/done ID - Mark food as consumed\n"
        "/delete ID - Delete a food entry\n"
        "/setdate ID YYYY-MM-DD - Correct an expiry date\n"
        "/help - Show help\n\n"
        "Dates estimated from a purchase are not verified expiry "
        "dates. Always check the product label and storage guidance."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)


async def list_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_private(update):
        return

    foods = list_foods(update.effective_user.id)
    if not foods:
        await update.message.reply_text(
            "Your inventory is empty. Send a grocery receipt photo to begin."
        )
        return

    lines = ["Your active groceries:"]
    for food in foods[:50]:
        expiry = food["expiry_date"] or "Date unknown"
        source = (
            "printed date"
            if food["date_source"] == "printed_on_receipt"
            else "ESTIMATE — please verify"
            if food["date_source"] == "rough_estimate"
            else "user corrected"
        )
        lines.append(
            f'#{food["id"]} {food["name"]} ({food["quantity"]})\n'
            f'  Date: {expiry} [{source}]'
        )

    if len(foods) > 50:
        lines.append("Showing the first 50 items.")

    await update.message.reply_text("\n\n".join(lines))


async def soon_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_private(update):
        return

    today = datetime.now(TZ).date()
    end = today + timedelta(days=3)
    foods = list_foods(update.effective_user.id)

    matching = []
    for food in foods:
        expiry = valid_date(food["expiry_date"])
        if expiry and today <= expiry <= end:
            matching.append(food)

    if not matching:
        await update.message.reply_text(
            "No dated items are due within the next 3 days. "
            "Items without dates are not included."
        )
        return

    lines = ["Expiring soon — verify each product label:"]
    for food in matching:
        lines.append(
            f'#{food["id"]} {food["name"]} — {food["expiry_date"]}'
        )

    await update.message.reply_text("\n".join(lines))


async def done_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_private(update):
        return

    if len(context.args) != 1 or not context.args[0].isdigit():
        await update.message.reply_text("Usage: /done ID")
        return

    success = mark_done(
        update.effective_user.id, int(context.args[0])
    )
    await update.message.reply_text(
        "Marked as consumed."
        if success else "Food entry not found or already completed."
    )


async def delete_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_private(update):
        return

    if len(context.args) != 1 or not context.args[0].isdigit():
        await update.message.reply_text("Usage: /delete ID")
        return

    success = delete_food(
        update.effective_user.id, int(context.args[0])
    )
    await update.message.reply_text(
        "Entry deleted." if success else "Food entry not found."
    )


async def setdate_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    if not is_private(update):
        return

    if len(context.args) != 2 or not context.args[0].isdigit():
        await update.message.reply_text(
            "Usage: /setdate ID YYYY-MM-DD"
        )
        return

    food_id = int(context.args[0])
    new_date = valid_date(context.args[1])

    if not new_date:
        await update.message.reply_text(
            "Invalid date. Use YYYY-MM-DD, for example 2026-10-20."
        )
        return

    success = change_expiry(
        update.effective_user.id, food_id, new_date
    )
    await update.message.reply_text(
        "Date updated and marked as user-corrected."
        if success else "Active food entry not found."
    )


# --------------------------------------------------
# 7. RECEIPT HANDLER
# --------------------------------------------------

async def receipt_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    if not is_private(update):
        return

    message = update.message
    await message.reply_text(
        "Reading your receipt. Please wait..."
    )

    try:
        if message.photo:
            telegram_file = await context.bot.get_file(
                message.photo[-1].file_id
            )
            image_bytes = bytes(await telegram_file.download_as_bytearray())

        elif message.document:
            mime_type = message.document.mime_type or ""
            if not mime_type.startswith("image/"):
                await message.reply_text(
                    "Please send a receipt as a photo or image file."
                )
                return

            if (
                message.document.file_size
                and message.document.file_size > 10 * 1024 * 1024
            ):
                await message.reply_text(
                    "That image is too large for this starter bot. "
                    "Please send a smaller image."
                )
                return

            telegram_file = await context.bot.get_file(
                message.document.file_id
            )
            image_bytes = bytes(
                await telegram_file.download_as_bytearray()
            )
        else:
            return

        result = read_receipt(image_bytes)
        today = datetime.now(TZ).date()
        items, receipt_date = normalize_items(result, today)

        if not items:
            await message.reply_text(
                "I couldn't identify grocery items. Try a clearer "
                "photo showing the full receipt."
            )
            return

        # First show a preview. Do not silently save every AI guess.
        context.user_data["pending_items"] = [
            {
                "name": item["name"],
                "quantity": item["quantity"],
                "category": item["category"],
                "purchase_date": item["purchase_date"].isoformat(),
                "expiry_date": (
                    item["expiry_date"].isoformat()
                    if item["expiry_date"] else None
                ),
                "date_source": item["date_source"],
            }
            for item in items
        ]

        lines = [
            f"Receipt date: {receipt_date.isoformat()}",
            "",
            "Please review these extracted items:",
        ]

        for index, item in enumerate(items, start=1):
            source = (
                "printed date"
                if item["date_source"] == "printed_on_receipt"
                else "ESTIMATED — verify"
            )
            expiry = (
                item["expiry_date"].isoformat()
                if item["expiry_date"] else "Unknown"
            )
            lines.append(
                f'{index}. {item["name"]} ({item["quantity"]})\n'
                f'   Date: {expiry} [{source}]'
            )

        lines.extend([
            "",
            "If the list looks right, send /confirm.",
            "To discard it, send /cancel.",
            "Correct important dates with /setdate after saving.",
        ])

        await message.reply_text("\n".join(lines))

    except Exception:
        logger.exception("Receipt processing failed")
        await message.reply_text(
            "I couldn't process that receipt. Check that the image "
            "is clear and your APIyi model supports image input. "
            "Try again or check the server logs."
        )


async def confirm_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    if not is_private(update):
        return

    pending = context.user_data.get("pending_items")
    if not pending:
        await update.message.reply_text(
            "There are no items waiting for confirmation. "
            "Send a receipt first."
        )
        return

    user_id = update.effective_user.id

    # Insert only the items the user has reviewed.
    for item in pending:
        add_food(
            user_id=user_id,
            name=item["name"],
            quantity=item["quantity"],
            category=item["category"],
            purchase_date=date.fromisoformat(item["purchase_date"]),
            expiry_date=(
                date.fromisoformat(item["expiry_date"])
                if item["expiry_date"] else None
            ),
            date_source=item["date_source"],
        )

    context.user_data.pop("pending_items", None)
    await update.message.reply_text(
        f"Saved {len(pending)} food entries. Use /list to view them."
    )


async def cancel_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    if not is_private(update):
        return

    context.user_data.pop("pending_items", None)
    await update.message.reply_text(
        "Pending receipt cancelled. Nothing was saved."
    )


# --------------------------------------------------
# 8. DAILY REMINDERS
# --------------------------------------------------

async def daily_reminders(context: ContextTypes.DEFAULT_TYPE):
    today = datetime.now(TZ).date()
    foods = []

    with db() as con:
        foods = con.execute("""
            SELECT user_id, id, name, expiry_date, date_source
            FROM foods
            WHERE status = 'active'
              AND expiry_date IS NOT NULL
              AND expiry_date <= ?
            ORDER BY user_id, expiry_date
        """, ((today + timedelta(days=3)).isoformat(),)).fetchall()

    by_user = {}
    for food in foods:
        expiry = valid_date(food["expiry_date"])
        if not expiry:
            continue

        # Do not send reminders for dates far in the past.
        if expiry < today - timedelta(days=7):
            continue

        by_user.setdefault(food["user_id"], []).append(food)

    for user_id, items in by_user.items():
        lines = ["FreshTrack daily reminder:"]
        for item in items[:30]:
            expiry = item["expiry_date"]
            if expiry < today.isoformat():
                timing = "date has passed"
            elif expiry == today.isoformat():
                timing = "date is today"
            else:
                timing = f"date: {expiry}"

            label = (
                "estimated"
                if item["date_source"] == "rough_estimate"
                else "label date / corrected"
            )
            lines.append(
                f'#{item["id"]} {item["name"]} — {timing} ({label})'
            )

        lines.append(
            "\nCheck the package label and storage conditions. "
            "A reminder date is not proof that food is safe or unsafe."
        )

        try:
            await context.bot.send_message(
                chat_id=user_id,
                text="\n".join(lines),
            )
        except Exception:
            logger.exception("Reminder failed for a user")


async def error_handler(update, context):
    logger.error("Unhandled Telegram error", exc_info=context.error)


# --------------------------------------------------
# 9. START THE BOT
# --------------------------------------------------

def main():
    init_db()

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("list", list_command))
    app.add_handler(CommandHandler("soon", soon_command))
    app.add_handler(CommandHandler("done", done_command))
    app.add_handler(CommandHandler("delete", delete_command))
    app.add_handler(CommandHandler("setdate", setdate_command))
    app.add_handler(CommandHandler("confirm", confirm_command))
    app.add_handler(CommandHandler("cancel", cancel_command))

    app.add_handler(
        MessageHandler(
            filters.PHOTO | filters.Document.IMAGE,
            receipt_handler,
        )
    )

    app.add_error_handler(error_handler)

    app.job_queue.run_daily(
        daily_reminders,
        time=datetime.strptime(
            f"{REMINDER_HOUR:02d}:00", "%H:%M"
        ).time().replace(tzinfo=TZ),
        name="daily_food_expiry_reminders",
    )

    logger.info("FreshTrack bot starting")
    app.run_polling()


if __name__ == "__main__":
    main()
