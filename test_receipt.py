"""
Telegram-free receipt test. Shows what the bot would track from a receipt image.
Usage:  python test_receipt.py path/to/receipt.jpg      (needs ANTHROPIC_API_KEY set)
"""
import asyncio, base64, json, os, sys

os.environ.setdefault("TELEGRAM_TOKEN", "unused")      # bot.py needs a value, but never uses it here
os.environ.setdefault("DB_PATH", "test_scratch.db")    # scratch file, safe to delete
import bot


async def main(path):
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(
        path.lower().rsplit(".", 1)[-1])
    if not mime:
        sys.exit("Please use a .jpg, .png or .webp file.")
    if os.path.getsize(path) > 5_000_000:
        sys.exit("That file is over 5 MB. Use a smaller photo.")
    img = base64.b64encode(open(path, "rb").read()).decode()
    print("Reading receipt (this takes a few seconds)...")
    r = await bot.LLM.messages.create(model=bot.MODEL, max_tokens=2000, messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": mime, "data": img}},
        {"type": "text", "text": bot.PROMPT}]}])
    raw = r.content[0].text
    print("\n--- Raw reply from the AI ---\n" + raw)
    data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    bought = bot.pick_date(data.get("receipt_date"), bot.DEFAULT_TZ)
    print(f"\n--- What the bot would do (purchase date used: {bought:%d %b %Y}) ---")
    for it in data.get("items", []):
        p = bot.plan(it, bought)
        if p:
            flag = "" if p["sure"] else "  ❓ unsure"
            print(f"TRACK  {p['name']} x{p['qty']} ({p['storage']}) eat by ~{p['expiry']}{flag}")
        else:
            print(f"SKIP   {it.get('name')}  (pantry, not perishable, or unrecognised)")


if len(sys.argv) != 2:
    sys.exit("Usage: python test_receipt.py path/to/receipt.jpg")
asyncio.run(main(sys.argv[1]))
