import csv
import json
import traceback
from datetime import datetime
from pathlib import Path

import discord.ext.commands
import pandas as pd
import pytz
from discord.ext import commands


def set_spelling(onoff):
    Path("nazi").write_text("on" if onoff else "off")


class AdminCog(commands.Cog):
    def __init__(self, bot: discord.ext.commands.Bot):
        self.bot = bot

    @commands.command(hidden=True)
    @commands.is_owner()
    async def msggc(self, ctx, guild: int, channel: int, *, words):
        await self.bot.get_guild(guild).get_channel(channel).send(words)

    @commands.command(hidden=True)
    @commands.is_owner()
    async def cleanup(self, ctx, user_id, channel_id=None, limit=5):
        if channel_id is None:
            channels = ctx.guild.text_channels
        else:
            channels = [ctx.guild.get_channel(int(channel_id))]
        count = 0
        for channel in channels:
            print("Cleaning up ", channel.name)

            def pred(msg):
                return msg.author.id == int(user_id)

            try:
                async for msg in channel.history(limit=limit).filter(pred):
                    try:
                        await msg.delete()
                        count += 1
                    except Exception as e:
                        print(f'failed to delete: {e}')
                        pass
            except:
                print('Unable to access', channel.name)
        print(f"Cleaned up {count} messages")

    @commands.command()
    @commands.is_owner()
    async def nazi(self, ctx, s):
        """
        Enable spelling nazi mode
        """
        enabled = s.lower() == "on"
        set_spelling(enabled)
        await ctx.send(f"Spelling Nazi mode {'' if enabled else 'dis'}engaged")

    @commands.command(hidden=True)
    @commands.is_owner()
    async def dump(self, ctx, status='old'):
        guild = self.bot.get_guild(self.bot.ADL_GUILD_ID)
        data = []
        meta = {
            'channels': {},
            'authors': {}
        }
        fname = 'log.csv'

        if status == 'old':
            meta = json.loads(Path(f"dump_meta.json").read_text())
            # only fetch from the last record
            try:
                old_rows = pd.read_csv(fname)
                last_date = datetime.fromtimestamp(old_rows.timestamp.max())
            except:
                last_date = None
        else:
            last_date = None

        print("Dumping posts after ", last_date)
        for channel in guild.text_channels:
            try:
                print(f"\tDumping {channel}")
                chan_messages = await channel.history(limit=5000000, oldest_first=True, after=last_date).flatten()
                with open(fname, 'a') as out_file:
                    writer = csv.DictWriter(out_file, fieldnames=[
                        'timestamp', 'channel', 'author', 'msg'])
                    for m in chan_messages:
                        meta['channels'][m.channel.id] = m.channel.name
                        if hasattr(m.author, 'nick') and m.author.nick is not None:
                            name = m.author.nick
                        else:
                            name = m.author.name
                        meta['authors'][m.author.id] = name
                        row = {
                            'channel': m.channel.id,
                            'author': m.author.id,
                            'timestamp': m.created_at.replace(tzinfo=pytz.UTC).timestamp(),
                            'msg': m.content
                        }
                        writer.writerow(row)
            except Exception as e:
                traceback.print_exc()
                print("failed to get history for ", channel)

        Path(f'dump_meta.json').write_text(json.dumps(meta, indent=4))
        print(f"Wrote")

    @commands.command(hidden=True)
    async def permcheck(self, ctx):
        print("\n".join(str(x)
                        for x in ctx.message.channel.guild.me.guild_permissions))

    @commands.command(hidden=True)
    @commands.is_owner()
    async def reload(self, ctx, cog: str):
        self.bot.reload_extension(f"cogs.{cog}")


def setup(ctx):
    ctx.add_cog(AdminCog(ctx))
