import re

import discord
from discord.ext import commands

from valve import rcon


class BhopCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    def do_rcon(self, cmd):
        addr = ('bhop.rip', 27015)
        pw = self.bot.key_store['rcon_pw']
        rcon.RCONMessage.ENCODING = 'utf-8'
        return rcon.execute(addr, pw, cmd)

    @commands.command()
    async def nextmap(self, ctx):
        """
        Shows the nextmap at bhop.rip
        """
        nm = self.do_rcon('nextmap')
        await ctx.send(nm)

    @commands.command()
    async def players(self, ctx):
        """
        Shows the current players of bhop.rip
        """
        players = re.findall(r'^players\s+:\s(.*)$', self.do_rcon('status'), re.M)[0]
        await ctx.send("**[SM]** Players: " + players)

    @commands.command()
    async def bhop(self, ctx):
        """
        Shows server details of bhop.rip
        """
        players = re.findall(r'^hostname:\s(.*)$', self.do_rcon('status'), re.M)[0]
        await ctx.send("**[SM]** " + players)

    @commands.command()
    async def currentmap(self, ctx):
        """
        Shows the current map at bhop.rip
        """
        map = re.findall(r'^map\s+:\s(\w+) ', self.do_rcon('status'), re.M)[0]
        await ctx.send("**[SM]** Current map is: " + map)

    @commands.Cog.listener("on_message")
    async def ff(self, message: discord.Message):
        if message.content.strip() == "ff":
            await message.channel.send("**[SM] Friendly fire is disabled.**")


def setup(ctx):
    ctx.add_cog(BhopCog(ctx))
