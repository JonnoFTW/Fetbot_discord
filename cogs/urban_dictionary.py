import discord
from discord.ext import commands

import aiohttp


class UrbanDictionaryCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def ud(self, ctx, word, pos=0):
        async with ctx.typing(), aiohttp.ClientSession() as session:
            async with session.get("https://api.urbandictionary.com/v0/define", params={'term': word}) as res:
                data = (await res.json())["list"]
                try:
                    entry = data[pos]
                    # await ctx.send(f"UrbanDictionary definition of {word}: {entry['definition']}\n👍{entry['thumbs_up']}👎{entry['thumbs_down']} {entry['permalink']}")
                    embed = discord.Embed(
                        title=entry['word'],
                        colour=0x006064,
                        url=entry['permalink'],
                    )
                    embed.add_field(name=f"**Definition**", value=entry['definition'])
                    embed.set_footer(text=f"👍{entry['thumbs_up']}👎{entry['thumbs_down']}")
                    await ctx.send(embed=embed)
                except IndexError:
                    await ctx.send("No definition at that position")


def setup(ctx):
    ctx.add_cog(UrbanDictionaryCog(ctx))
