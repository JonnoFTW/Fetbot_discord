from discord.ext import commands


class StatsCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(aliases=['profile'])
    async def statsme(self, ctx):
        """ Show your stats"""
        await ctx.send(f"{ctx.author.mention} User stats http://45.248.76.3:5000/u/{ctx.author.id}")

    @commands.command()
    async def stats(self, ctx):
        await ctx.send(f"Channel stats http://45.248.76.3:5000/c/{ctx.channel.id}")


def setup(ctx):
    ctx.add_cog(StatsCog(ctx))
