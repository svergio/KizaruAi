import discord
from discord.ext import commands
TOKEN = 'OTQwNjA0MzExODk1OTQ5MzQy.YgJ0Bw.aUnjv1T2fWBDHvW9Vz2bujzQ-y8'
bot = commands.Bot(command_prefix='Ai')


@bot.event
async def on_ready():
    print("Bot is ready")

@bot.command(pass_context=True)
async def help(ctx, arg):
    await ctx.send("тестовая залупа кста хрочешь поговрить попробуй Talk")

bot.run(TOKEN)

