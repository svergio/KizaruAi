import argparse
import os

import discord
from discord.ext import commands

from kizaru.checkpoint import get_device, load_model
from kizaru.mood import load_mood, mood_source, predict_mood
from kizaru.pic import discord_chunks, render_pic, resolve_mode
from kizaru.sample import evaluate

DISCORD_LIMIT = 1800


async def replied_content(ctx):
    reference = ctx.message.reference
    if reference is None or reference.message_id is None:
        return ""
    message = reference.resolved
    if message is None:
        try:
            message = await ctx.channel.fetch_message(reference.message_id)
        except (discord.NotFound, discord.HTTPException):
            return ""
    return message.content or ""


async def image_bytes(message):
    for attachment in message.attachments:
        name = attachment.filename.lower()
        content_type = attachment.content_type or ""
        if content_type.startswith("image/") or name.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
            return await attachment.read()
    return None


async def pic_bytes(ctx):
    data = await image_bytes(ctx.message)
    if data:
        return data
    reference = ctx.message.reference
    if reference is None or reference.message_id is None:
        return None
    message = reference.resolved
    if message is None:
        try:
            message = await ctx.channel.fetch_message(reference.message_id)
        except (discord.NotFound, discord.HTTPException):
            return None
    if message is None:
        return None
    return await image_bytes(message)


def build_bot(model, char_to_idx, idx_to_char, device, mood=None, prefix="Ai"):
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(
        command_prefix=(prefix, prefix.lower(), "!"),
        intents=intents,
        case_insensitive=True,
    )

    @bot.event
    async def on_ready():
        print(f"Bot is ready as {bot.user}")

    @bot.command(name="talk")
    async def talk(ctx, *, start: str = "Я"):
        text = evaluate(
            model,
            char_to_idx,
            idx_to_char,
            start_text=start,
            prediction_len=280,
            temp=0.4,
            device=device,
            max_lines=6,
        )
        await ctx.send(text[:DISCORD_LIMIT])

    @bot.command(name="mood")
    async def mood_cmd(ctx, *, text: str = ""):
        if mood is None:
            await ctx.send("Модель настроения не передана. Запусти бота с --mood checkpoints/mood.pt")
            return
        source = mood_source(await replied_content(ctx), text)
        if not source:
            await ctx.send("Ответь на сообщение командой AiMood, либо напиши текст после команды.")
            return
        mood_model, vocab = mood
        label, score = predict_mood(mood_model, vocab, source)
        snippet = " ".join(source.split())
        if len(snippet) > 140:
            snippet = snippet[:140] + "..."
        await ctx.send(f"Настроение: {label} ({score:.0%})\n{snippet}"[:DISCORD_LIMIT])

    @bot.command(name="pic")
    async def pic(ctx, mode: str = "simple", cells: int = 24):
        if mode.isdigit():
            cells = int(mode)
            mode = "simple"
        try:
            mode_name = resolve_mode(mode)
        except ValueError:
            await ctx.send("Режим: AiPic 1 для лайна или AiPic 2 для простого среднего цвета.")
            return
        data = await pic_bytes(ctx)
        if not data:
            await ctx.send("Кинь картинку или ответь на сообщение с картинкой.")
            return
        cells = max(8, min(int(cells), 40))
        text = render_pic(data, mode=mode_name, cells=cells)
        for chunk in discord_chunks(text):
            await ctx.send(chunk[:DISCORD_LIMIT])

    return bot


def parse_args():
    parser = argparse.ArgumentParser(description="Discord bot that loads an exported TextRNN")
    parser.add_argument("model", help="Path to the Kizaru text .pt file")
    parser.add_argument("--mood", default=None, help="Path to the mood .pt file")
    parser.add_argument("--token", default=os.environ.get("DISCORD_TOKEN", ""))
    parser.add_argument("--prefix", default="Ai")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    model, char_to_idx, idx_to_char, meta = load_model(args.model, device=device)
    print(f"loaded text {args.model} vocab={len(char_to_idx)} loss={meta.get('mean_loss')}")
    mood = None
    if args.mood:
        mood_model, vocab, mood_meta = load_mood(args.mood, device=device)
        mood = (mood_model, vocab)
        print(f"loaded mood {args.mood} vocab={len(vocab)} test_acc={mood_meta.get('notebook_test_acc')}")
    token = args.token.strip()
    if not token:
        raise SystemExit("Pass --token or set DISCORD_TOKEN")
    bot = build_bot(model, char_to_idx, idx_to_char, device, mood=mood, prefix=args.prefix)
    bot.run(token)


if __name__ == "__main__":
    main()
