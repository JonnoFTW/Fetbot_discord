import asyncio
import random
from datetime import datetime, timedelta

import discord.ext.commands
import pandas as pd
from discord import utils
from discord.ext import commands


class GladiatorsCog(commands.Cog):
    GLADIATOR_ROLE = 992000264028557352
    GLADIATOR_CHANNEL = 991998965342019684

    def __init__(self, bot: discord.ext.commands.Bot):
        self.bot = bot
        self.bot.loop.create_task(self.gladiator_task())

        self.GLADIATORS = {
            'WEAPON_CURRENT': None,
            'WEAPON_HOLDER': None
        }

    def get_gladiators(self, exclude=[]):
        print("gladiators adl_guild_id", self.bot.ADL_GUILD_ID, self.bot.get_guild(self.bot.ADL_GUILD_ID))

        role = utils.get(self.bot.get_guild(self.bot.ADL_GUILD_ID).roles, name='Gladiator')
        return [m for m in role.members if m.id not in exclude]

    @commands.command()
    @commands.is_owner()
    async def count_old(self, ctx, days: int):
        df = pd.read_csv('log.csv')
        year_ago = (datetime.now() - timedelta(days=days)).timestamp()
        recent_chatters = set(df.loc[df['timestamp'] > year_ago].author.unique())
        non_chatters = []
        users_to_ban = []
        role = utils.get(ctx.guild.roles, name='Gladiator')
        changes = []
        existing_gladiators = [u.id for u in self.get_gladiators()]
        for member in ctx.guild.members:
            if member.id not in recent_chatters:
                non_chatters.append(member)
                if member.display_name and member.id not in existing_gladiators:
                    users_to_ban.append(member.display_name)
                    print(f"Moving {member.display_name} to the colosseum")
                changes.append(member.edit(roles=[role], reason="Volunteered for the colosseum"))
        await asyncio.gather(*changes)

        print(f"non-chatters={len(non_chatters)} recent_chatters = {len(recent_chatters)}, {users_to_ban}")
        await ctx.send(f"There are {len(non_chatters)} who have not chatted in the last {days} days out of {len(ctx.guild.members)} current members")

    def make_weapon(self):
        adj = ['rusty', 'clean', 'spiky', 'shiny', 'jeweled', 'broken', 'perfect', 'large', 'small', 'big', 'tiny', 'dirty', 'disgusting', 'cursed', 'blessed', 'haunted', 'evil', 'soul-stealing', 'obliterating', 'super']
        noun = ['gun', 'knife', 'sword', 'rifle', 'spear', 'dagger', 'sabre', 'glaive', 'knuckles', 'pistol', 'frying pan', 'rock', 'club', 'shotgun', 'trident', 'kitchen knife', 'garrot', 'chainsaw', 'machete']
        return f"{random.choice(adj)} {random.choice(noun)}"

    async def say_gladiators(self, msg):
        return await self.bot.get_channel(self.GLADIATOR_CHANNEL).send(msg)

    @commands.command()
    @commands.has_role(GLADIATOR_ROLE)
    async def use(self, ctx, *, name):
        if self.GLADIATORS['WEAPON_HOLDER'] == ctx.author.id and self.GLADIATORS['WEAPON_CURRENT'] == name:
            other_gladiators = self.get_gladiators(exclude=[ctx.author.id])
            enemy = random.choice(other_gladiators)
            await self.say_gladiators(f"{ctx.author.display_name} has used the {self.GLADIATORS['WEAPON_CURRENT']} against {enemy.mention}")
            await enemy.kick(reason="Died in combat")
            await self.say_gladiators(f"{enemy.mention} has been killed. {len(self.get_gladiators())} remain")
            self.GLADIATORS['WEAPON_CURRENT'] = None
            self.GLADIATORS['WEAPON_HOLDER'] = None
            await self.reload_weapon()
        else:
            await self.say_gladiators(f"You don't hold that weapon")

    @commands.command()
    @commands.has_role(GLADIATOR_ROLE)
    async def pickup(self, ctx, *, name):
        if self.GLADIATORS['WEAPON_HOLDER'] is None:
            if name == self.GLADIATORS['WEAPON_CURRENT']:
                self.GLADIATORS['WEAPON_HOLDER'] = ctx.author.id
                await self.say_gladiators(f"{ctx.author.mention} please type `.use {self.GLADIATORS['WEAPON_CURRENT']}` to kill someone. You have 10 minutes to comply")
        else:
            await self.say_gladiators(f"{self.GLADIATORS['WEAPON_CURRENT']} is already held by someone else")

    async def reload_weapon(self):
        self.GLADIATORS['WEAPON_CURRENT'] = self.make_weapon()
        weapon = self.GLADIATORS['WEAPON_CURRENT']
        self.GLADIATORS['WEAPON_HOLDER'] = None
        channel = self.bot.get_channel(self.GLADIATOR_CHANNEL)
        return await channel.send(f"Dropping the {weapon} on the ground. First person to type `.pickup {weapon}` will kill a random Gladiator. If nobody picks it up I will kill someone myself")

    async def gladiator_task(self):
        await asyncio.sleep(5)
        while True:
            # drop a weapon
            users = self.get_gladiators()
            if len(users) == 1:
                await self.say_gladiators(f"{users[0].mention} congratulations on winning the colosseum squid-game purge! You may use the rest of the server, you now have the survivor role.")
                survivor_role = utils.get(self.bot.get_guild(self.bot.ADL_GUILD_ID).roles, name='Survivor')
                await users[0].edit(roles=[survivor_role])
                return
            if len(users) == 0:
                await asyncio.sleep(60 * 60)
                continue
            await self.reload_weapon()
            await asyncio.sleep(60 * 10)
            # if nobody has picked up the weapon, ban a random person
            if self.GLADIATORS['WEAPON_HOLDER'] is None:
                enemy = random.choice(users)
                await self.say_gladiators(f"Nobody has picked up the {self.GLADIATORS['WEAPON_CURRENT']} yet. I'm killing {enemy.mention} myself")
                await self.bot.get_guild(self.bot.ADL_GUILD_ID).kick(enemy, reason="Failed to comply with orders")
                await self.say_gladiators(f"{enemy.mention} has been killed. {len(self.get_gladiators())} remain")
                self.GLADIATORS['WEAPON_CURRENT'] = None
                self.GLADIATORS['WEAPON_HOLDER'] = None


def setup(ctx):
    ctx.add_cog(GladiatorsCog(ctx))
