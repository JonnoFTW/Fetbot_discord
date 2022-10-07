import pickle
from pathlib import Path

import discord
from discord.ext.commands import Cog


class SpellingCog(Cog):
    def __init__(self, bot):
        self.bot = bot
        with open('mistakes.pkl', 'rb') as infile:
            self.mistakes = pickle.load(infile)

    @Cog.listener("on_message")
    async def spelling(self, message: discord.Message):
        mcl = message.content.lower()

        check = Path("nazi").read_text().strip() == "on"

        if check:
            for word in mcl.split():
                if word in self.mistakes:
                    await message.channel.send(f"{message.author.mention} it's spelt '{self.mistakes[word]}'")


def setup(ctx):
    ctx.add_cog(SpellingCog(ctx))
