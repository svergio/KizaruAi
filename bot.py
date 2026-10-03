import os
from pathlib import Path

from train import get_device, load_checkpoint
from kizaru_rnn.generate import evaluate

try:
    import discord
    from discord.ext import commands
except ImportError as exc:
    raise SystemExit("Install discord.py: pip install -r requirements.txt") from exc


DEFAULT_CHECKPOINT = Path("checkpoints/kizaru.pt")


def load_model():
    checkpoint = Path(os.environ.get("KIZARU_CHECKPOINT", DEFAULT_CHECKPOINT))
    if not checkpoint.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint}. Train first: python train.py"
        )
    device = get_device()
    model, char_to_idx, idx_to_char, payload = load_checkpoint(checkpoint, device=device)
    print(f"loaded {checkpoint} vocab={len(char_to_idx)} loss={payload.get('mean_loss')}")
    return model, char_to_idx, idx_to_char, device


def build_bot(model, char_to_idx, idx_to_char, device):
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix=("Ai", "ai", "!"), intents=intents)

    @bot.event
    async def on_ready():
        print(f"Bot is ready as {bot.user}")

    @bot.command(name="talk", aliases=("Talk",))
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
        if len(text) > 1800:
            text = text[:1800]
        await ctx.send(text)

    @bot.command(name="helpme")
    async def helpme(ctx):
        await ctx.send("Напиши `AiTalk <строка>` или `!talk <строка>` — модель допишет дальше.")

    return bot


def main():
    token = os.environ.get("DISCORD_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set DISCORD_TOKEN in the environment or .env")
    model, char_to_idx, idx_to_char, device = load_model()
    bot = build_bot(model, char_to_idx, idx_to_char, device)
    bot.run(token)


if __name__ == "__main__":
    main()
