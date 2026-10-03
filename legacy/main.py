import discord
from discord.ext import commands
TOKEN = ''  # revoked; do not commit Discord tokens
bot = commands.Bot(command_prefix='Ai')


@bot.event
async def on_ready():
    print("Bot is ready")

@bot.command(pass_context=True)
async def help(ctx, arg):
    await ctx.send("тестовая залупа кста хрочешь поговрить попробуй Talk")

bot.run(TOKEN)

