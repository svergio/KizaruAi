import argparse
import os

import discord
from discord.ext import commands

from kizaru.checkpoint import get_device, load_model
from kizaru.sample import evaluate

DISCORD_LIMIT = 1800


def build_bot(model, char_to_idx, idx_to_char, device, prefix="Ai"):
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

    return bot


def parse_args():
    parser = argparse.ArgumentParser(description="Discord bot that loads an exported TextRNN")
    parser.add_argument("model", help="Path to the trained .pt file")
    parser.add_argument("--token", default=os.environ.get("DISCORD_TOKEN", ""))
    parser.add_argument("--prefix", default="Ai")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    model, char_to_idx, idx_to_char, meta = load_model(args.model, device=device)
    print(f"loaded {args.model} vocab={len(char_to_idx)} loss={meta.get('mean_loss')}")
    token = args.token.strip()
    if not token:
        raise SystemExit("Pass --token or set DISCORD_TOKEN")
    bot = build_bot(model, char_to_idx, idx_to_char, device, prefix=args.prefix)
    bot.run(token)


if __name__ == "__main__":
    main()
