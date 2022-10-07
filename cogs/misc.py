import json
import subprocess
import time
from datetime import datetime, timedelta
from itertools import cycle
from pathlib import Path
import random
import re

import discord
import humanize
import numpy as np
import pytz
from discord.ext import commands

from syllables import syllable_count

jokes = json.loads(Path('jokes.json').read_text())


class MiscCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(prefix='')
    async def asl(self, ctx):
        """a/s/l"""
        places = ['sa', 'hawaii', 'israel', 'nigeria', 'aus', 'cali', 'nyc', 'nsw', 'vic',
                  'fl', 'uk', 'france', 'russia', 'germany', 'japan', 'china', 'nz', 'uganda']
        await ctx.send('/'.join([str(np.random.randint(8, 30)), random.choice(['m', 'f']), random.choice(places)]))

    @commands.command()
    async def flip(self, ctx):
        """Flip a coin"""
        await ctx.send(f"A coin is flipped: {random.choice(['heads', 'tails'])}")

    @commands.command()
    async def roulette(self, ctx, user_a: discord.User, user_b: discord.User):
        """
        Nominate 2 users to play russian roulette
        """
        it = cycle([user_b, user_a])
        c_user = next(it)
        msg = f"Comrades {user_a.display_name} and {user_b.display_name} have volunteered to play Russian Roulette"
        msg += f"\nA single round is loaded into the revolver and placed against {c_user.display_name}'s temple 😨🔫"
        await ctx.send(msg)
        pos = np.random.randint(0, 6)
        current_pos = 0
        while True:
            time.sleep(1)
            msg = "I pull the trigger..."
            if pos == current_pos:
                await ctx.send(msg + f" **BANG** . {c_user.display_name}'s body slumps to the floor as blood splatters your friends")
                break
            else:
                c_user = next(it)
                await ctx.send(msg + f" **click**. The revolver is placed against {c_user.display_name}'s head")
            current_pos += 1

    @commands.command()
    async def fortune(self, ctx):
        """
        Give you a fortune
        """
        msg = subprocess.check_output(["fortune", "-s"]).decode('utf-8').strip()
        await ctx.send(msg)

    @commands.command()
    async def roll(self, ctx, *args):
        """
        Roll n x sided dice
        """
        num = 1
        sides = 6
        try:
            num = int(args[0])
            sides = int(args[1])
        except:
            pass
        num = np.clip([num], 1, 16)[0]
        sides = np.max([2, sides])
        dice = "die" if num == 1 else "dice"
        res = ", ".join([str(x) for x in np.random.random_integers(1, sides, num)])
        await ctx.send(f"Rolls {num} sexy {sides} sided {dice}: {res}")

    @commands.command()
    async def joke(self, ctx):
        """
        Tells a funny joke
        """
        await ctx.send(np.random.choice(jokes))

    @commands.command()
    async def ts(self, ctx):
        """" Show the current timestamp"""
        t = ctx.message.created_at.replace(tzinfo=pytz.UTC)
        await ctx.send(f"{t.timestamp()} {t.tzinfo}")

    @commands.command()
    async def thetime(self, ctx):
        """The current time in Adelaide, Australia"""
        tz = pytz.timezone("Australia/Adelaide")
        await ctx.send(re.sub(r" 0(\d)", r" \g<1>", f"The time is: {datetime.now().astimezone(tz):%I:%M%p %A %d %B, %Y}"))

    @commands.command(name='time')
    async def timeleft(self, ctx):
        """Time left before the end of the day in Adelaide/Australia"""
        tz = pytz.timezone("Australia/Adelaide")
        mins = ((datetime.now().astimezone(tz) + timedelta(days=1)).replace(hour=0,
                                                                            minute=0, second=0) - datetime.now().astimezone(tz)).total_seconds() / 60
        await ctx.send(f'There are {int(mins)} minutes left in the day')

    @commands.command(name='twitter')
    async def get_twitter(self, ctx, user):
        """
        Show the last tweet of a given twitter user
        """
        try:
            t = self.bot.twitter_api.GetUserTimeline(screen_name=user)[0]
            await ctx.send(
                "@%s %s: %s" % (t.user.screen_name, humanize.naturaldelta(datetime.utcnow() - datetime.strptime(t.created_at, '%a %b %d %H:%M:%S +0000 %Y')), "https://twitter.com/{}/status/{}".format(t.user.screen_name, t.id_str)))
        except Exception as e:
            await ctx.send(f"Error Getting twitter: {e}")

    @commands.command()
    async def syllables(self, ctx, word):
        """
        How many syllables a word has
        """
        await ctx.send(f"{word} has {syllable_count(word)} syllables")


def setup(ctx):
    ctx.add_cog(MiscCog(ctx))
