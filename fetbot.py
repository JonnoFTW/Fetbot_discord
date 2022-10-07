#!/usr/bin/env python3
import sys
import asyncio
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import discord

import matplotlib.pylab as plt
import pytz
import requests
import twitter
import yaml

from discord.ext import commands

"""
Run with:
~/markov-discord$ PYTHONPATH=. ./fetbot.py

"""

ADL_GUILD_ID = 570212775499137024

with open('keys.yaml', 'r') as infile:
    key_store = yaml.safe_load(infile)

USER_AGENT = {
    'User-Agent': 'Mozilla/5.0 (X11; Fedora; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'}

intents = discord.Intents.default()
intents.members = True

bot = commands.Bot(command_prefix='.', intents=intents)
bot.key_store = key_store
bot.ADL_GUILD_ID = ADL_GUILD_ID
bot.USER_AGENT = USER_AGENT

data_store = 'data.json'
lock = asyncio.Lock()

bot.twitter_api = twitter.Api(
    key_store['consumer_key'],
    key_store['consumer_secret'],
    key_store['access_token_key'],
    key_store['access_token_secret']
)


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'Noto Sans CJK JP',
    'Helvetica Neue', 'Helvetica'
]


async def get_db():
    async with lock:
        with open(data_store, 'r') as fh:
            out = json.load(fh)
            # print(f"Read: {json.dumps(out)}")
            return out


async def write_db(data):
    async with lock:
        with open(data_store, 'w') as fh:
            # print(f"Writing: {json.dumps(data)}")
            json.dump(data, fh)


bot.get_db = get_db
bot.write_db = write_db


@bot.event
async def on_ready():
    print(f"Logged on")
    await bot.change_presence(activity=discord.Game(name="Cock/Ball Torture"))


@bot.event
async def on_member_join(member):
    channel = member.guild.get_channel(570216721512792105)
    if channel:
        await channel.send(f'{member.mention}: please remember that asking (including sending DMs) for drugs or links to underground drug servers will result in an instant ban')


@bot.event
async def on_message(message):
    if type(message.channel) is discord.DMChannel:
        name = "DirectMessage"
        gid = ""
    else:
        name = message.channel.name
        gid = message.guild.id
    log_msg = f"({datetime.now()}) ({gid}) #{name}(chanid={message.channel.id})(userid={message.author.id})\t{message.author.name}\t: {message.content}"
    # print("Message channel type:", message.channel, type(message.channel))
    if isinstance(message.channel, discord.channel.TextChannel):
        row_dict = {
            'channel': message.channel.id,
            'author': message.author.id,
            'timestamp': message.created_at.replace(tzinfo=pytz.UTC).timestamp(),
            'msg': message.content
        }
        with open('log.csv', 'a') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=[
                'timestamp', 'channel', 'author', 'msg'])
            writer.writerow(row_dict)
        try:
            res = requests.post(
                "http://localhost:5000/update", json=[row_dict])
            # print(f"Updating chanstat: ", res.status_code)
        except Exception as e:
            print(e)

    print(log_msg)
    with open('log.txt', 'a') as out_log:
        out_log.write(log_msg)
        out_log.write("\n")
    if message.author == bot.user:
        return

    mcl = message.content.lower()
    if any(x in mcl for x in ("drug", "underground")) and any(x in mcl for x in ("link", "server")):
        await message.add_reaction("<:brainlet:736521294144602113>")
        await message.reply('Asking for links to drug servers is a bannable offense')
    if any(x in re.sub(r'[\W_]', '', mcl) for x in ["mynudesforfree", "videoswithyouforfree", "wanttoseemynudes", "sharemyhotpic"]):
        try:
            await message.delete()
        except:
            print("couldn't delete message")
        print(f"Kicking {message.author.name} for saying: '{message.content}'")
        try:
            await message.author.kick(reason="TRY GETTING A RESERVATION AT DORSIA NOW YOU STUPID FUCKING BASTARD")
        except:
            print("failed to kick", message.author)
        return



extensions = [
    'cogs.admin',
    'cogs.bhop',
    'cogs.fetish',
    'cogs.gladiators',
    'cogs.imggen',
    'cogs.lastfm',
    'cogs.misc',
    'cogs.spelling',
    'cogs.stats',
    'cogs.urban_dictionary',
    'cogs.weather',
]

if __name__ == "__main__":
    token = Path('tokens')
    for ext in extensions:
        try:
            bot.load_extension(ext)
            print(f"Loaded extension {ext}")
        except Exception as err:
            print(f"Failed to load extension {ext}: ", err)
    while True:
        try:
            bot.run(token.read_text().strip())
        except KeyboardInterrupt:
            sys.exit("Closing")
        except:
            pass
