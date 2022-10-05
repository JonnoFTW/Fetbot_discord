#!/usr/bin/env python3
import sys
import asyncio
import csv
import json
import logging
import os
import pickle
import random
import re
import subprocess
import time
import traceback
from datetime import datetime, timedelta
from ftplib import FTP
from glob import glob
from io import BytesIO
from itertools import cycle
from pathlib import Path
from urllib.parse import urlparse

import discord
from discord import utils
import ngrok
import humanize
import aiohttp
import matplotlib.dates as mdates
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pytz
import requests
import seaborn as sns
import twitter
import yaml
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor
from discord.ext import commands
from emoji import get_emoji_regexp, emojize
from nltk.metrics import distance
from sklearn.cluster import KMeans
from valve import rcon

from syllables import syllable_count


do_nsfw_check = False
poster_chance = 0.02
if do_nsfw_check:
    tf.get_logger().setLevel(logging.ERROR)
    tf.autograph.set_verbosity(3)
    nsfw_model = get_model()

"""
Run with:
~/markov-discord$ TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=~/opennsfw ./fetbot.py

"""
ADL_GUILD_ID = 570212775499137024

with open('keys.yaml', 'r') as infile:
    key_store = yaml.safe_load(infile)
intents = discord.Intents.default()
intents.members = True
bot = commands.Bot(command_prefix='.', intents=intents)
jokes = json.loads(Path('jokes.json').read_text())
data_store = 'data.json'
lock = asyncio.Lock()
USER_AGENT = {
    'User-Agent': 'Mozilla/5.0 (X11; Fedora; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'}

api = twitter.Api(
    key_store['consumer_key'],
    key_store['consumer_secret'],
    key_store['access_token_key'],
    key_store['access_token_secret']
)


def set_spelling(onoff):
    Path("nazi").write_text("on" if onoff else "off")


def _spelling():
    return Path("nazi").read_text().strip() == "on"


with open('mistakes.pkl', 'rb') as infile:
    _mistakes = pickle.load(infile)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP',
                                   'Helvetica Neue', 'Helvetica']


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


with open('fetishes.json', 'r') as fh:
    fetishes = json.load(fh)


def strip(msg):
    return re.sub(r'\s+', ' ', re.sub(r"[;-_,]", '', msg.lower()))


def get_canonical(msg):
    s = msg
    if s in fetishes:
        return s
    for k, v in fetishes.items():
        if k == s or s in v:
            return k


def find_similar(fet):
    thresh = 3
    out = []
    for f, v in fetishes.items():
        if any(distance.edit_distance(fet, x) <= thresh for x in [f] + v):
            out.append(f)
    return ', '.join(out)


async def status_task():
    while True:
        await asyncio.sleep(180)
        await bot.change_presence(activity=discord.Game(name=random.choice(list(fetishes.keys())).title()))


@bot.check
async def restrict_to(ctx):
    return ctx.message.channel.id in {570213862285115393, 570216721512792105, 570248053567782922, 648525630857674762} or not hasattr(ctx.message, 'server')


@bot.event
async def on_ready():
    print(f"Logged on")
    await bot.change_presence(activity=discord.Game(name="Cock/Ball Torture"))
    bot.loop.create_task(status_task())
    bot.loop.create_task(gladiator_task())


@bot.command()
@commands.is_owner()
async def msggc(ctx, guild: int, channel: int,*, words):
    await bot.get_guild(guild).get_channel(channel).send(words)


def get_gladiators(exclude=[]):
    role = utils.get(bot.get_guild(ADL_GUILD_ID).roles, name='Gladiator')
    return [m for m in role.members if m.id not in exclude]
    
@bot.command()
@commands.is_owner()
async def count_old(ctx, days: int):
    df = pd.read_csv('log.csv')
    year_ago = (datetime.now() - timedelta(days=days)).timestamp()
    recent_chatters = set(df.loc[df['timestamp'] > year_ago].author.unique())
    non_chatters = []
    users_to_ban = []
    role = utils.get(ctx.guild.roles, name='Gladiator')
    changes = []
    existing_gladiators = [u.id for u in get_gladiators()]
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

GLADIATOR_ROLE = 992000264028557352
GLADIATOR_CHANNEL = 991998965342019684
GLADIATORS = {
    'WEAPON_CURRENT': None,
    'WEAPON_HOLDER': None
}

def make_weapon():
    adj = ['rusty', 'clean', 'spiky', 'shiny', 'jeweled', 'broken', 'perfect', 'large', 'small', 'big', 'tiny', 'dirty', 'disgusting', 'cursed' , 'blessed','haunted', 'evil', 'soul-stealing', 'obliterating', 'super']
    noun = ['gun', 'knife', 'sword', 'rifle','spear', 'dagger', 'sabre', 'glaive', 'knuckles', 'pistol', 'frying pan', 'rock', 'club', 'shotgun', 'trident', 'kitchen knife','garrot', 'chainsaw', 'machete']
    return f"{random.choice(adj)} {random.choice(noun)}"


async def say_gladiators(msg):
    return await bot.get_channel(GLADIATOR_CHANNEL).send(msg)

@bot.command()
@commands.has_role(GLADIATOR_ROLE)
async def use(ctx, *, name):
    if GLADIATORS['WEAPON_HOLDER'] == ctx.author.id and GLADIATORS['WEAPON_CURRENT'] == name:
        other_gladiators = get_gladiators(exclude=[ctx.author.id])
        enemy = random.choice(other_gladiators)
        await say_gladiators(f"{ctx.author.display_name} has used the {GLADIATORS['WEAPON_CURRENT']} against {enemy.mention}")
        await enemy.kick(reason="Died in combat")
        await say_gladiators(f"{enemy.mention} has been killed. {len(get_gladiators())} remain")
        GLADIATORS['WEAPON_CURRENT'] = None
        GLADIATORS['WEAPON_HOLDER'] = None
        await reload_weapon()
    else:
        await say_gladiators(f"You don't hold that weapon")
    
@bot.command()
@commands.has_role(GLADIATOR_ROLE)
async def pickup(ctx, *, name):
    if GLADIATORS['WEAPON_HOLDER'] is None:
        if name == GLADIATORS['WEAPON_CURRENT']:
            GLADIATORS['WEAPON_HOLDER'] = ctx.author.id
            await say_gladiators(f"{ctx.author.mention} please type `.use {GLADIATORS['WEAPON_CURRENT']}` to kill someone. You have 10 minutes to comply")
    else:
        await say_gladiators(f"{GLADIATORS['WEAPON_CURRENT']} is already held by someone else")


async def reload_weapon():
    GLADIATORS['WEAPON_CURRENT'] = make_weapon()
    weapon = GLADIATORS['WEAPON_CURRENT']
    GLADIATORS['WEAPON_HOLDER'] = None
    channel = bot.get_channel(GLADIATOR_CHANNEL)
    return await channel.send(f"Dropping the {weapon} on the ground. First person to type `.pickup {weapon}` will kill a random Gladiator. If nobody picks it up I will kill someone myself")


async def gladiator_task():
    while True:
        # drop a weapon
        users = get_gladiators()
        if len(users) == 1:
            await say_gladiators(f"{users[0].mention} congratulations on winning the colosseum squid-game purge! You may use the rest of the server, you now have the survivor role.")
            survivor_role = utils.get(bot.get_guild(ADL_GUILD_ID).roles, name='Survivor')
            await users[0].edit(roles=[survivor_role])
            return
        if len(users) == 0:
            await asyncio.sleep(60*60)
            continue
        await reload_weapon()
        await asyncio.sleep(60*10)
        # if nobody has picked up the weapon, ban a random person
        if GLADIATORS['WEAPON_HOLDER'] is None:
            enemy = random.choice(users)
            await say_gladiators(f"Nobody has picked up the {GLADIATORS['WEAPON_CURRENT']} yet. I'm killing {enemy.mention} myself")
            await bot.get_guild(ADL_GUILD_ID).kick(enemy, reason="Failed to comply with orders")
            await say_gladiators(f"{enemy.mention} has been killed. {len(get_gladiators())} remain")
            GLADIATORS['WEAPON_CURRENT'] = None
            GLADIATORS['WEAPON_HOLDER'] = None
        

@bot.command()
@commands.is_owner()
async def nazi(ctx, s):
    """
    Enable spelling nazi mode
    """
    enabled = s.lower() == "on"
    set_spelling(enabled)
    await ctx.send(f"Spelling Nazi mode {'' if enabled else 'dis'}engaged")


@bot.command()
@commands.is_owner()
async def add_fetish(ctx, *args):
    """
    Add a fetish definition
    """
    f = ' '.join(args)
    if f not in fetishes:
        fetishes[f] = []
        with open('fetishes.json', 'w') as fh:
            json.dump(fetishes, fh)

        await ctx.send("Done")
    else:
        await ctx.send("Already in there")


@bot.command(prefix='')
async def asl(ctx):
    """a/s/l"""
    places = ['sa', 'hawaii', 'israel', 'nigeria', 'aus', 'cali', 'nyc', 'nsw', 'vic',
              'fl', 'uk', 'france', 'russia', 'germany', 'japan', 'china', 'nz', 'uganda']
    await ctx.send('/'.join([str(np.random.randint(8, 30)), random.choice(['m', 'f']), random.choice(places)]))


@bot.command()
async def flip(ctx):
    """Flip a coin"""
    await ctx.send(f"A coin is flipped: {random.choice(['heads', 'tails'])}")


@bot.command()
async def roulette(ctx, user_a: discord.User, user_b: discord.User):
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


@bot.command()
async def fortune(ctx):
    """
    Give you a fortune
    """
    msg = subprocess.check_output(["fortune", "-s"]).decode('utf-8').strip()
    await ctx.send(msg)


@bot.command()
async def roll(ctx, *args):
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


@bot.command()
async def add(ctx, *args):
    """
    Record that you have a fetish
    """
    data = await get_db()
    stripped = strip(' '.join(args))
    fet = get_canonical(stripped)
    if not fet:
        s = "No such fetish exists"
        suggestions = find_similar(stripped)
        if suggestions:
            s += ". Perhaps you meant: " + suggestions
        await ctx.send(s)
        return
    data[str(ctx.author.id)] = list(
        set(data.get(str(ctx.author.id), []) + [fet]))
    # print(data)
    await ctx.send("Done" if random.random() > 0.05 else "Yes, Daddy")
    await write_db(data)


@bot.command()
@commands.is_owner()
async def pool(ctx, user_a: discord.User, user_b: discord.User):
    """
    Record a victory in pool
    """
    data = await get_db()
    if 'pool' not in data:
        data['pool'] = []
    data['pool'].append([user_a.id, user_b.id])
    await write_db(data)
    await ctx.send(f"{user_a.name} beat {user_b.name} in CBT")


@bot.command()
@commands.is_owner()
async def set_kink(ctx):
    """
    Doesn't do anything
    """
    msg = await ctx.bot.get_channel(570223709022060547).fetch_message(570432571591360532)
    await msg.add_reaction(emoji='👺')


#  for reaction in msg.reactions:
#      users = await reaction.users().flatten()
#      for i in users:
#          print(f"{reaction.emoji}=>{i}")


@bot.command()
async def leaderboard(ctx):
    """
   Not available
    """
    data = await get_db()
    matches = data.get('pool')
    if not matches:
        await ctx.send("No games played")
        return
    out = "Coming soon"
    #    players =  set([x.split(" ") for x in matches])
    await ctx.send(out)


@bot.command()
async def table(ctx):
    """
    Show a table of kink compatability scores
    """
    data = await get_db()
    for k in list(data.keys()):
        if not k.isdigit() or ctx.guild.get_member(int(k)) is None:
            del data[k]
    tbl = []
    users = []
    for x in data.keys():
        row = []
        uname = ctx.bot.get_user(int(x))

        users.append(uname.display_name if uname is not None else "?")
        for y in data.keys():
            row.append(iou(data[x], data[y]))
        tbl.append(row)
    # oldcmap = 'YlGnBu'
    arry = np.array(tbl)
    mask = np.zeros_like(arry, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    ax = plt.subplot()
    sns.heatmap(arry, annot=True, linewidth=0.5, cmap="cool",  # fontdict={'fontname': 'NotoEmoji-merged'},
                xticklabels=users, yticklabels=users, square=True, mask=mask, ax=ax)
    # for text_obj in ax.get_xticklabels():
    #    text_obj.set_fontname('Uni Sans')
    # for text_obj in ax.get_yticklabels():
    #    text_obj.set_fontname('Uni Sans')
    ax.set_title(
        r"Kink Kompatibility $\left( \frac{A \cap B}{A \cup B} \right)$")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45,
                       verticalalignment='center')
    buff = BytesIO()
    plt.savefig(buff, format='png')
    buff.seek(0)
    plt.cla()
    plt.clf()
    # plt.savefig("table.png")
    fname = f'table_{time.time()}.png'
    file = discord.File(buff, filename=fname)
    embed = discord.Embed()
    embed.set_image(url=f'attachment://{fname}')
    await ctx.send(file=file, embed=embed)


@bot.command()
async def bomold(ctx):
    """
    Show the rain radar for Adelaide
    """
    now = datetime.now()
    url = f"http://www.bom.gov.au/radar/IDR643.gif?t={time.time()}"
    data = requests.get(url, headers=USER_AGENT).content
    path = Path('radar.gif')
    path.write_bytes(data)
    file = discord.File("radar.gif")
    embed = discord.Embed()
    embed.set_image(url='attachment://radar.gif')
    await ctx.send(file=file, embed=embed)

def get_guild_city(ctx, key):
    default = {'radar': 'Adelaide (Buckland Park)', 'weather': 'adelaide'}
    with open('cities.json') as in_file:
        data = json.load(in_file)
        return data.get(str(ctx.guild.id), default)[key]


@bot.command()
async def bom(ctx, *, site=None):
    radars = {
       "Brewarrina":"IDR933",
       "Canberra (Captains Flat)":"IDR403",
       "Grafton":"IDR283",
       "Hillston":"IDR943",
       "Moree":"IDR533",
       "Namoi (Blackjack Mountain)":"IDR693",
       "Newcastle":"IDR043",
       "Norfolk Island":"IDR623",
       "Sydney":"IDR713",
       "Wagga Wagga":"IDR553",
       "Wollongong (Appin)":"IDR033",
       "Yeoval":"IDR963",
       "Alice Springs":"IDR253",
       "Darwin (Berrimah)":"IDR633",
       "Gove":"IDR093",
       "Katherine (Tindal)":"IDR423",
       "Warruwi":"IDR773",
       "Bowen":"IDR243",
       "Brisbane (Mt Stapylton)":"IDR663",
       "Cairns":"IDR193",
       "Emerald":"IDR723",
       "Gladstone":"IDR233",
       "Greenvale":"IDR743",
       "Gympie (Mt Kanigan)":"IDR083",
       "Longreach":"IDR563",
       "Mackay":"IDR223",
       "Marburg":"IDR503",
       "Mornington Island":"IDR363",
       "Mount Isa":"IDR753",
       "Taroom":"IDR983",
       "Townsville (Hervey Range)":"IDR733",
       "Warrego":"IDR673",
       "Weipa":"IDR783",
       "Willis Island":"IDR413",
       "Adelaide (Buckland Park)":"IDR643",
       "Adelaide (Sellicks Hill)":"IDR463",
       "Ceduna":"IDR333",
       "Mt Gambier":"IDR143",
       "Woomera":"IDR273",
       "Hobart (Mt Koonya)":"IDR763",
       "Hobart Airport":"IDR373",
       "N.W. Tasmania (West Takone)":"IDR523",
       "Bairnsdale":"IDR683",
       "Melbourne":"IDR023",
       "Mildura":"IDR973",
       "Rainbow":"IDR953",
       "Yarrawonga":"IDR493",
       "Albany":"IDR313",
       "Broome":"IDR173",
       "Carnavon":"IDR053",
       "Dampier":"IDR153",
       "Esperance":"IDR323",
       "Geraldton":"IDR063",
       "Giles":"IDR443",
       "Halls Creek":"IDR393",
       "Kalgoorlie":"IDR483",
       "Learmonth":"IDR293",
       "Newdegate":"IDR383",
       "Perth (Serpentine)":"IDR703",
       "Port Hedland":"IDR163",
       "South Doodlakine":"IDR583",
       "Watheroo":"IDR793",
       "Wyndham":"IDR073"
    }
    if site is None:
        try:
            site = get_guild_city(ctx, 'radar')
        except:
            pass
    else:
        if site not in radars:
            await ctx.send(f'No such radar site {site}. Perhaps you meant: {get_suggestions(radars, site)}')
            return

    rid = radars.get(site, radars['Adelaide (Buckland Park)'])
    prefix = "http://www.bom.gov.au/products/radar_transparencies/"
    fnames = ["IDR.legend.0.png", f"{rid}.background.png",
              f"{rid}.topography.png", f"{rid}.range.png", f"{rid}.locations.png"]
    out_name = 'radar_animated.gif'
    async with ctx.typing():
        image = None
        for f in fnames:
            p = Path(f)
            if not p.exists():
                res = requests.get(f"{prefix}{f}", headers=USER_AGENT)
                p.write_bytes(res.content)
            if image is None:
                image = Image.open(f).convert('RGBA')
            else:
                foreground = Image.open(f).convert('RGBA')
                image.paste(foreground, (0, 0), foreground)
        images = []
        range_cover = Image.open(fnames[3]).convert("RGBA")
        location_cover = Image.open(fnames[4]).convert("RGBA")

        # page = "http://www.bom.gov.au/products/IDR643.loop.shtml"
        # page_text = requests.get(page, headers=USER_AGENT).text
        # rain_frames = re.findall(r"theImageNames\[\d+] = \"/radar/(.*)\"", page_text)
        font = ImageFont.truetype('/home/jonno/.fonts/NotoSans-Bold.ttf', 22)
        with FTP('ftp.bom.gov.au') as ftp:
            ftp.login('anonymous', 'guest')
            prefix = '/anon/gen/radar'
            rain_frames = [f for f in ftp.nlst(prefix) if f'{rid}.T' in f]
            for f in rain_frames:
                p = Path('radar/' + f.split('/')[-1])
                # print(f'Fetching {f}')
                if not p.exists():
                    with p.open('wb') as radar_frame_f:
                        ftp.retrbinary("RETR " + f, radar_frame_f.write)
                new_frame = image.copy()
                mask = Image.open(str(p)).convert("RGBA")
                for cover in [mask, range_cover, location_cover]:
                    new_frame.paste(cover, (0, 0), cover)
                frame_time = datetime.strptime(f.split(
                    '.')[2] + '+0000', '%Y%m%d%H%M%z').astimezone(pytz.timezone('Australia/Adelaide'))
                draw = ImageDraw.Draw(new_frame)
                text = frame_time.strftime('%b %d %I:%M %p')
                w, h = font.getsize(text)
                x, y = 125, 500
                m = 4
                draw.rounded_rectangle(
                    (x - m, y - m, x + w + m, y + h + m), fill=(0, 0, 0, 168), radius=7)
                draw.text((x, y), text, fill=(255, 255, 255), font=font)
                images.append(new_frame)
                # p.unlink()
        images[0].save(out_name, append_images=images[1:],
                       duration=500, loop=0, save_all=True, optimize=True, disposal=1, include_color_table=True)
        file = discord.File(out_name)
        embed = discord.Embed()
        embed.set_image(url=f'attachment://{out_name}')
        await ctx.send(file=file, embed=embed)


@bot.command()
async def remove(ctx, *args):
    """
    Remove a fetish
    """
    data = await get_db()
    fet = get_canonical(' '.join(args))
    if not fet:
        await ctx.send("No such fetish exists")
        return
    aid = str(ctx.author.id)
    if fet in data[aid]:
        data[aid].remove(fet)
        await ctx.send("Done" if random.random() > 0.05 else "Yes, Daddy")
        await write_db(data)
    else:
        await ctx.send("You don't have that fetish")


def iou(a, b):
    aset = set(x for x in a)
    bset = set(x for x in b)
    return len(aset.intersection(bset)) / float(len(aset.union(bset)))


@bot.command()
async def find(ctx):
    """
    Find your most compatible kinkster
    """
    data = await get_db()
    if 570226229379072020 in [r.id for r in ctx.author.roles]:
        await ctx.send("18+ only")
        return
    if len(data) <= 1:
        await ctx.send("There's nobody else to compare yourself to")
        return
    score = max(
        sorted([
            (
                iou(data.get(str(ctx.author.id), tuple()),
                    data.get(str(other), tuple())), other
            ) for other in data.keys() if other != str(ctx.author.id) and other.isdigit() and ctx.message.guild.get_member(int(other))
        ], reverse=True)
    )
    if score[0] == 0:
        await ctx.send(f"You have no match with anyone. Try using .add <fetish name>")
    else:
        await ctx.send(f"{ctx.author.mention} your best match is <@{score[1]}> with score {round(score[0], 3)}")


@bot.command()
@commands.is_owner()
async def dump(ctx, status='old'):
    guild = bot.get_guild(ADL_GUILD_ID)
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


@bot.command(name='random')
async def get_random(ctx):
    """
    Tell you a random fetish
    """
    await ctx.send("Why not try " + random.choice(list(fetishes.keys())) + "?")


@bot.command(name='show_all')
async def show_all(ctx):
    """
    Give you the link to the master list of fetishes
    """
    await ctx.send("All available fetishes are listed here: http://45.248.76.3:5000/fetishes")  # https://gist.github.com/JonnoFTW/7788169e843a685e37628d9cd8a0be6b")

sites = {}

def load_sites():
    with open('bom.dat') as f:
        for line in f:
            state, ids, location = line.split(' ', 2)
            location = location.title().strip()
            sites[location] = ids.split('.')

load_sites()


def get_suggestions(d, thing):
    return ', '.join([k for k in d.keys() if distance.edit_distance(thing, k) < 3])

@bot.command()
async def weather(ctx, *, site=None):
    """
    Weather details for selected site
    """
    if site is None:
        site = get_guild_city(ctx, 'weather')

    lower_sites = {k.lower(): v for k, v in sites.items()}
    if site.lower() in lower_sites:
        ids_id, site_id = lower_sites[site.lower()]
        print("site", site, site_id)
    else:
        await ctx.send(f"No such site '{site}' exists. Perhaps you meant: {get_suggestions(sites, site)}")
        return

    url = f"http://www.bom.gov.au/fwo/{ids_id}/{ids_id}.{site_id}.json"
    o = requests.get(url, headers=USER_AGENT).json()['observations']['data'][0]
    out = {
        #            'City': o['name'],
        'Temp(°C)': o['air_temp'],
        'Wind(km/h)': o['wind_spd_kmh'],
        'Rain(mm)': o['rain_trace'],
        'Humidity(%)': o['rel_hum'],
        'Wind Dir': o['wind_dir'],
        #'Visibility(km)': o['vis_km'],
        'Updated': o['local_date_time']
    }

    embed = discord.Embed(
        title=o['name'],
        colour=0x006064,
        url=f"http://www.bom.gov.au/products/{ids_id}/IDS{ids_id}.{site_id}.shtml"
    )

    for k, v in out.items():
        embed.add_field(name=f"**{k}**", value=f"\n{v}")
    await ctx.send(embed=embed)


@bot.command()
async def temps(ctx, site=None, field='air_temp', other_field=None):
    """
    Shows a chart of the recent temperatures in Adelaide
    Specify the city name in quote marks eg. .temps "coffs harbour" apparent_t
    """
    fields = {
        "wind_spd_kmh": "Wind Speed (km/h)",
        # "vis_km": "Visibility (km)",
        "rel_hum": "Relative Humidity (%)",
        "press": "Pressure (hPa)",
        "dewpt": "Dew Point (°C)",
        "gust_kmh": "Gusts (km/h)",
        "air_temp": "Air Temperature (°C)",
        "rain_trace": "Rain since 9am (mm)",
        "cloud_base_m": "Cloud Base (m)",
        "apparent_t": "Apparent Temperature (°C)",
        "delta_t": "Wet Bulb Depression (°C)",
    }
    
    if site is None:
        site = get_guild_city(ctx, 'weather')
        
    
        
    lower_sites = {k.lower(): v for k, v in sites.items()}
    if site.lower() in lower_sites:
        ids_id, site_id = lower_sites[site.lower()]
        print("site", site, site_id)
    else:
        await ctx.send(f"No such site '{site}' exists. Perhaps you meant: {get_suggestions(sites, site)}")
        return

    async with ctx.typing():
        url = f"http://reg.bom.gov.au/fwo/{ids_id}/{ids_id}.{site_id}.json"
        resp = requests.get(url)
        if not resp.ok:
        
            print(url)
            await ctx.send("Something went wrong")
            return
        all_data = resp.json()
        data = all_data['observations']['data']
        if field not in fields:
            await ctx.send(f"Field must be one of {', '.join(['**' + f + '**: ' + v for f, v in fields.items()])}")
            return

        df = pd.DataFrame(data)
        df['rain_trace'] = pd.to_numeric(df['rain_trace'])
        # df['vis_km'] = pd.to_numeric(df['vis_km'])
        df.set_index(pd.to_datetime(df['local_date_time_full']), inplace=True)

        ax = df[field].plot(label=fields[field])
        ax.set_ylabel(fields[field])
        lines = ax.get_lines()
        title = fields[field]

        if other_field is not None:
            df[other_field].plot(secondary_y=True, label=fields[other_field])
            title = title + " and\n " + fields[other_field]
            ax.right_ax.set_ylabel(fields[other_field])
            lines += ax.right_ax.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], loc='upper left')
        plt.title(f"{title} In {all_data['observations']['header'][0]['name']}")
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%I %p"))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.set_xlabel('Date')
        for text in ax.get_xminorticklabels():
            text.set_rotation(70)
            text.set_fontsize('x-small')
        for text in ax.get_xmajorticklabels():
            text.set_rotation(70)
            text.set_ha('center')
        ax.grid(visible=True, which='major', linewidth=2)
        ax.grid(visible=True, which='minor', linewidth=0.5)
        ax.grid(which='major', axis='y', visible=True, linewidth=0.5)
        buff = BytesIO()
        plt.savefig(buff, format='png')
        buff.seek(0)
        plt.cla()
        plt.clf()
        fname = f'temps_{time.time()}.png'
        file = discord.File(buff, filename=fname)
        embed = discord.Embed()
        embed.set_image(url=f'attachment://{fname}')
        await ctx.send(file=file, embed=embed)


emo_re = get_emoji_regexp()


def fit_text(txt, draw, font_path, imwidth, size=128, padding=32):
    print("txt has emoji: ", emo_re.search(txt), txt)
    while 1:
        if emo_re.search(txt):
            try:
                #                imfont = ImageFont.truetype("/home/jonno/.fonts/Symbola.ttf",size=size, encoding='unic')
                imfont = ImageFont.truetype(
                    "/home/jonno/.fonts/NotoEmoji-merged.ttf", size=size, encoding='unic')
            except OSError:
                size = size - 2
                continue
        else:
            imfont = ImageFont.truetype(font_path, size=size)
        tw = draw.textsize(txt, imfont)
        if tw[0] < imwidth - 32 or size < 16:
            return imfont, tw
        else:
            size = size - 2


with open('colours.json', 'r') as infile:
    colours = json.load(infile)
all_colours = []
for br in colours.values():
    for col_list in br.values():
        all_colours.extend([sRGBColor.new_from_rgb_hex(c) for c in col_list])


def hex2rgb(s):
    h = s.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def get_colour(im, y_start):
    region = im.crop(
        (0, y_start, im.size[0], y_start + 64)).filter(ImageFilter.BLUR)
    arr = np.array(region)
    avg = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    # bright background, use dark colour font
    use_dark = (avg * np.array([0.299, 0.587, 0.114])).sum() > 186
    font_br = "light" if use_dark else "dark"
    key = np.random.choice(
        [k for k in colours.keys() if font_br in colours[k]])
    cols = colours[key][font_br]
    print("Use dark=", use_dark)
    return hex2rgb(np.random.choice(cols))


def get_colour_dist(im, y_start, ts, x_start=16):
    region = np.array(
        im.crop((x_start, y_start, x_start + ts[0], y_start + ts[1])))
    # take histogram of region , find colour furthest from the most common
    region = region.reshape((region.shape[0] * region.shape[1], 3))
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(region)
    dc_rgb = kmeans.cluster_centers_[0]
    dc_lab = convert_color(sRGBColor(*dc_rgb, is_upscaled=True), LabColor)
    # for each color in all_colors, get the max, might be made parallel
    res = max(all_colours, key=lambda a: delta_e_cie2000(
        dc_lab, convert_color(a, LabColor)))
    return res.get_upscaled_value_tuple()


@bot.command()
async def permcheck(ctx):
    print("\n".join(str(x)
                    for x in ctx.message.channel.guild.me.guild_permissions))


@bot.command(name='poster')
async def poster(ctx, arg1="", arg2="", arg3="", *args, **kwargs):
    """
    Generate a cool posters. Use quote marks to separate lines eg: .poster "first line" "" "third line"
    """
    print(f"poster args= '{arg1}' '{arg2}' '{arg3}' {args} ")
    async with ctx.typing():
        mentions = {}
        try:
            for uid in ctx.message.raw_mentions:
                member = await ctx.guild.fetch_member(uid)
                mention = f'<@!{uid}>'
                arg1 = arg1.replace(mention, '@' + member.display_name)
                arg2 = arg2.replace(mention, '@' + member.display_name)
                arg3 = arg3.replace(mention, '@' + member.display_name)
        except Exception as e:
            traceback.print_exc()
        arg1 = await commands.clean_content(fix_channel_mentions=True, use_nicknames=True).convert(ctx, arg1)
        arg2 = await commands.clean_content(fix_channel_mentions=True, use_nicknames=True).convert(ctx, arg2)
        arg3 = await commands.clean_content(fix_channel_mentions=True, use_nicknames=True).convert(ctx, arg3)
        if kwargs.get('dall_e'):
            im_buf, _, _, _, _ = await get_dall_e_img(ctx, f'{arg1}, {arg2}, {arg3} --steps 120', kwargs['dall_e'])
        else:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://source.unsplash.com/random?nature") as resp:
                    if resp.status == 200:
                        im_buf = BytesIO(await resp.read())
        im = Image.open(im_buf)
        basewidth = 960
        wpercent = (basewidth / float(im.size[0]))
        hsize = int((float(im.size[1]) * float(wpercent)))
        im = im.resize((basewidth, hsize))
        font_dir = "/home/jonno/.fonts/*.ttf"
        fonts = [f for f in glob(font_dir) if re.search(
            r"Noto(Sans|Serif|SerifDisplay)-", f)]
        draw = ImageDraw.Draw(im)
        if arg1:
            # replace any mentions with the server name
            fonta, ts = fit_text(
                arg1, draw, np.random.choice(fonts), basewidth, 106)
            y_start = 16
            #draw.rectangle([16-4, y_start-4, ts[0]+16, ts[1]+4], fill=bg_col)
            draw.text((16, y_start), arg1, get_colour_dist(
                im, y_start, ts), font=fonta, stroke_width=3, stroke_fill="white")
        if arg2:
            fontb, ts = fit_text(
                arg2, draw, np.random.choice(fonts), basewidth, 96)
            y_start = im.size[1] / 2 - 12
            #draw.rectangle([16-4, y_start-4, ts[0]+4, ts[1]+4], fill=bg_col)
            draw.text((16, y_start), arg2, get_colour_dist(
                im, y_start, ts), font=fontb, stroke_width=3, stroke_fill="black")
        if arg3:
            fontc, ts = fit_text(
                arg3, draw, np.random.choice(fonts), basewidth, 76)
            y_start = im.size[1] - 128
            #draw.rectangle([16-4, y_start-4, ts[0]+4, ts[1]+4], fill=bg_col)
            draw.text((16, y_start), arg3, get_colour_dist(
                im, y_start, ts), font=fontc, stroke_width=3, stroke_fill="black")

        buff = BytesIO()
        im.save(buff, format='png', quality=95)
        buff.seek(0)
        fname = f'poster_{time.time()}.png'
        file = discord.File(buff, filename=fname)
        embed = discord.Embed()
        if check_haiku(" ".join([arg1, arg2, arg3])):
            embed.title = "Haiku"
            embed.description = ctx.message.author.mention
        embed.set_image(url=f'attachment://{fname}')
        await ctx.send(file=file, embed=embed)


@bot.command()
async def set_lastfm(ctx, username):
    """
    Let me know your last.fm username
    """
    db = await get_db()
    if 'lfm' not in db:
        db['lfm'] = {}
    db['lfm'][str(ctx.author.id)] = username
    await write_db(db)
    await ctx.send(f"Set {ctx.author.name}'s lastfm username to '{username}'")


@bot.command(name='np')
async def now_playing(ctx, user=''):
    """
    Fetch the song you are currently playing from last.fm
    """
    key = key_store['last_fm_api']
    async with ctx.typing():
        if not user:
            db = await get_db()
            user = db.get('lfm', {}).get(str(ctx.author.id), None)
            if user is None:
                await ctx.send(f"No lastfm name set, use .set_lastfm <username>")
                return
        print("Using username", user)
        data = requests.get("http://ws.audioscrobbler.com/2.0/", {
            'method': 'user.getrecenttracks', 'user': user, 'format': 'json', 'api_key': key}).json()
        song = data['recenttracks']['track'][0]
        msg = f"**{user}** {'is now listening' if '@attr' in song else 'last listened'} to \"*{song['name']}*\" by {song['artist']['#text']} from *{song['album']['#text']}*\n{song['url']}"
        tags = []
        try:
            tags_data = requests.get("http://ws.audioscrobbler.com/2.0/", {
                'method': 'track.getInfo', 'format': 'json', 'api_key': key,
                'artist':song['artist']['#text'], 'track': song['name']}).json()
            print("got tags", tags_data)
            for tag in tags_data['track']['toptags']['tag']:
                tags.append(tag['name'])
        except Exception as e:
            print(e)
            pass
        
        try:
            link = song['image'][2]['#text']
            im_data = requests.get(
                ('http://' if not link.startswith('http') else '') + song['image'][2]['#text']).content
            buff = BytesIO(im_data)
            file = discord.File(buff, filename="cover.jpg")
            embed = discord.Embed()
            embed.set_image(url=f'attachment://cover.jpg')
            embed.description = f"{ctx.author.mention}, tags: {', '.join(tags)}"
            await ctx.send(msg, file=file, embed=embed)
        except Exception as e:
            print(e)
            await ctx.send(msg)


@bot.command()
async def genre(ctx):
    """Come up with your new favourite genre"""
    prefixes = ['enterprise', '', 'post', 'indie', 'avant-garde', 'nautical', 'break', 'wub', 'chip', 'vintage', 'classic', 'virtuosic', 'death', 'instrumental', 'british', 'industrial', 'thrash', 'japanese', 'J', 'K', 'acoustic',
                'progressive', 'power', 'glam', 'melodic', 'new wave', 'german', 'gothic', 'symphonic', 'grind', 'synth',
                'minimal', 'psychedelic', 'brutal', 'sexy', 'easy listening', 'christian', 'anime', 'stoner', 'comedy', 'sad', 'christmas', 'neo', 'russian', 'finnish', 'summer', 'underground', 'dream', 'pagan', 'minimal', 'ambient', 'nu',
                'speed', 'contemporary', 'alt', 'acid', 'english', 'kvlt', 'cult', 'mu', 'raw', 'norwegian', 'viking', 'porn']
    suffixes = ['core', '', 'step', 'groove', 'noise']
    gens = ['folk', 'ambient', 'electronica', 'funk', 'hip-hop', 'dance', 'pop', 'trance', 'indie', 'soul', 'hard', 'lounge', 'blues', 'classical', 'grunge', '/mu/core', 'emo', 'rap', 'rock',
            'punk', 'alternative', 'nautical', 'electro', 'swing', 'screamo', 'jazz', 'reggae', 'metal', 'classical', 'math', 'nerd', 'country', 'western', 'dub', "drum 'n' bass", 'celtic', 'shoegaze']
    x = random.choice(prefixes)
    if x:
        x += '-'
        if random.randint(0, 2) == 1:
            x += random.choice(prefixes) + '-'
    x += random.choice(gens)
    if random.randint(0, 3) == 1:
        x += random.choice(suffixes)
    await ctx.send(x)


@bot.command(name='list')
async def list_fets(ctx):
    """
    DMs you your fetishes
    """
    data = await get_db()
    await ctx.author.send("Your fetishes are " + (', '.join(data.get(str(ctx.author.id), []))))


@bot.command()
async def score(ctx, *, member: discord.User):
    """
    Get the score for you and someone else
    """
    data = await get_db()
    if str(member.id) in data:
        score = iou(data.get(str(ctx.author.id), []),
                    data.get(str(member.id), []))
        await ctx.send(f"{ctx.author.mention} your compatibility with {member.name} is {score}")
    else:
        await ctx.send(f"{ctx.author.mention}: {member.name} is not a kinkster")


@bot.command()
async def nextmap(ctx):
    """
    Shows the nextmap at bhop.rip
    """
    nm = do_rcon('nextmap')
    await ctx.send(nm)


@bot.command()
async def players(ctx):
    """
    Shows the current players of bhop.rip
    """
    players = re.findall(r'^players\s+:\s(.*)$', do_rcon('status'), re.M)[0]
    await ctx.send("**[SM]** Players: " + players)


@bot.command(aliases=['profile'])
async def statsme(ctx):
    """ Show your stats"""
    await ctx.send(f"{ctx.author.mention} User stats http://45.248.76.3:5000/u/{ctx.author.id}")


@bot.command()
async def ts(ctx):
    """" Show the current timestamp"""
    t = ctx.message.created_at.replace(tzinfo=pytz.UTC)
    await ctx.send(f"{t.timestamp()} {t.tzinfo}")


@bot.command()
async def stats(ctx):
    await ctx.send(f"Channel stats http://45.248.76.3:5000/c/{ctx.channel.id}")


@bot.command()
async def bhop(ctx):
    """
    Shows server details of bhop.rip
    """
    players = re.findall(r'^hostname:\s(.*)$', do_rcon('status'), re.M)[0]
    await ctx.send("**[SM]** " + players)


@bot.command()
async def currentmap(ctx):
    """
    Shows the current map at bhop.rip
    """
    map = re.findall(r'^map\s+:\s(\w+) ', do_rcon('status'), re.M)[0]
    await ctx.send("**[SM]** Current map is: " + map)


@bot.command()
async def thetime(ctx):
    """The current time in Adelaide, Australia"""
    tz = pytz.timezone("Australia/Adelaide")
    await ctx.send(re.sub(r" 0(\d)", r" \g<1>", f"The time is: {datetime.now().astimezone(tz):%I:%M%p %A %d %B, %Y}"))


@bot.command(name='time')
async def timeleft(ctx):
    """Time left before the end of the day in Adelaide/Australia"""
    tz = pytz.timezone("Australia/Adelaide")
    mins = ((datetime.now().astimezone(tz) + timedelta(days=1)).replace(hour=0,
                                                                        minute=0, second=0) - datetime.now().astimezone(tz)).total_seconds() / 60
    await ctx.send(f'There are {int(mins)} minutes left in the day')


@bot.command()
async def joke(ctx):
    """
    Tells a funny joke
    """
    j = np.random.choice(jokes)
    await ctx.send(j)


def do_rcon(cmd):
    addr = ('bhop.rip', 27015)
    pw = key_store['rcon_pw']
    rcon.RCONMessage.ENCODING = 'utf-8'
    return rcon.execute(addr, pw, cmd)


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

@bot.command()
async def syllables(ctx, word):
    """
    How many syllables a word has
    """
    await ctx.send(f"{word} has {syllable_count(word)} syllables")


def get_dall_ep():
    client = ngrok.Client(key_store['ngrok'])
    for tn in client.tunnels.list():
        if tn.metadata == "dall-e":
            return tn.public_url
    return None


def is_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


class DallError(Exception):
    pass


async def get_dall_e_img(ctx, msg, ep):
    """
    Return a buffer with the generated image
    ctx: discord message context
    msg: the message
    ep: the endpoint to call
    """
    async with aiohttp.ClientSession() as session:
        parts = re.split(r"--([A-Za-z]+)", ("--q " if "--q" not in msg else "") + msg)[1:]
        args = dict(zip(parts[0::2], [x.strip() for x in parts[1::2]]))
        err = False
        arg_funcs = {
            'steps': dict(out='ddim_steps', range=[1, 200], func=int),
            'W': dict(out='W', range=[64, 728], func=int),
            'H': dict(out='H', range=[64, 728], func=int),
            'scale': dict(out='scale', range=[1, 64], func=float),
            'strength': dict(out='strength', range=[0, 1], func=float),
            'seed': dict(out='seed', range=[-np.inf, np.inf], func=int),
        }
        ext = "jpg"
        if 'vid' in args:
            del args['vid']
            ext = 'webm'
        for arg, r in arg_funcs.items():
            if arg in args:
                try:
                    val = r['func'](args[arg])
                    if not r['range'][0] <= val <= r['range'][1]:
                        raise Exception()
                    del args[arg]
                    args[r['out']] = val
                except:
                    await ctx.send(f"Arg {arg} must be a {r['func'].__name__} in range {r['range']}, got {args[arg]}\nargs={args}")
                    err = True
                    break
        if err:
            raise ValueError("Bad args")
        if isinstance(ctx.channel, discord.channel.DMChannel):
            headers = {
                'X-Discord-User': ctx.message.author.name,
                'X-Discord-UserId': str(ctx.message.author.id),
                'X-Discord-Server': 'DM',
                'X-Discord-Channel': 'DM'
            }
        else:
            headers = {
                'X-Discord-User': ctx.message.author.display_name,
                'X-Discord-UserId': str(ctx.message.author.id),
                'X-Discord-Server': ctx.message.guild.name,
                'X-Discord-ServerId': str(ctx.message.guild.id),
                'X-Discord-Channel': ctx.channel.name,
                'X-Discord-ChannelId': str(ctx.channel.id)
            }
        start_t = int(time.time())
        if len(ctx.message.attachments) > 0:
            route = "img2img.jpg"
            formdata = aiohttp.FormData()
            formdata.add_field('file', BytesIO(await ctx.message.attachments[0].read()), filename='image.jpg')
            kwargs = {'data': formdata}
            method = session.post
        else:
            route = f"img.{ext}"
            kwargs = {}
            method = session.get
        async with method(f'{ep}/{route}', params=args, headers=headers, **kwargs) as resp:
            if resp.status == 200:
                buff = BytesIO(await resp.read())
                duration = str(round(time.time() - start_t, 2))
                return buff, resp, args, duration, ext
            else:
                resp_text = await resp.text()
                raise DallError(f"Dall-e service returned error ({resp.status}): {resp_text}")



@bot.command(aliases=["dream", "sd", "diffuse", "diffusion"])
async def dalle(ctx, *, msg):
    """
    will generate an image from prompt, if you upload an image, it will be generated on that
    Extra arguments are:
      --steps : number of steps of refinement, 10 is fast, 50 is okay, 150 is great
      --scale : adherence to the prompt, 7.5 is good
      --H     : height in pixels
      --W     : width in pixels
      --seed  : the seed, same seed with same prompt will make the same image
      --strength :  used for img2img, a float in range [0..1]
    """
    # put the message in the queue
    if not msg:
        await ctx.send("You must provide a prompt, see `.sd --help` for more")
        return
    endpoint = get_dall_ep()
    if not endpoint:
        await ctx.send("dream is not running")
        return
    try:
        buff, resp, args, duration, ext = await get_dall_e_img(ctx, msg, endpoint)
        fname = f"dalle.{ext}"
        buff.seek(0)
        file = discord.File(buff, filename=fname)
        embed = discord.Embed()
        embed.set_image(url=f'attachment://{fname}')
        embed.description = f"{ctx.message.author.mention}\n{args['q']}"
        if len(args) > 1:
            embed.add_field(name=f"Args", value=" ".join([f"{k}={v}" for k, v in args.items() if k != 'q']))
        embed.add_field(name=f"Took", value=duration + "secs")
        if 'X-SD-Seed' in resp.headers:
            embed.add_field(name='Seed', value=resp.headers['X-SD-Seed'])
        if ext == 'webm':
            embed = None
        await ctx.message.reply(file=file, embed=embed)
    except DallError as err:
        print(str(err))
        await ctx.message.reply("Error with service")


@bot.command()
async def ud(ctx, word, pos=0):
    async with ctx.typing():
        res = requests.get("https://api.urbandictionary.com/v0/define", params={'term': word})
        data = res.json()["list"]
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


@bot.command(name='cleanup')
@commands.is_owner()
async def cleanup(ctx, user_id, channel_id=None, limit=5):
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


@bot.command(name='twitter')
async def get_twitter(ctx, user):
    """
    Show the last tweet of a given twitter user
    """
    try:
        t = api.GetUserTimeline(screen_name=user)[0]
        await ctx.send("@%s %s: %s" % (t.user.screen_name, humanize.naturaltime(datetime.utcnow() - datetime.strptime(t.created_at, '%a %b %d %H:%M:%S +0000 %Y')), "https://twitter.com/{}/status/{}".format(t.user.screen_name, t.id_str)))
    except Exception as e:
        await ctx.send(f"Error Getting twitter: {e}")


async def handle_bot_mention(message):
    is_dm = type(message.channel) is discord.DMChannel
    if is_dm or message.channel.is_nsfw():
        if is_dm:
            is_female = random.choice([True, False])
        else:
            role_string = ' '.join([r.name.lower() for r in message.author.roles])
            role_ids = [r.id for r in message.author.roles]
            is_female = 570225951657689088 in role_ids or re.search(r'woman|girl|female|femme', role_string.lower())
        if is_female:
            name = "Mommy"
            s_name = "girl"
            b_part = "cunt"
        else:
            name = "Daddy"
            s_name = "boy"
            b_part = "cock"
        await message.channel.send(message.author.mention + " " + random.choice(["You can't talk to me", f"Harder, {name}", "Eyes closed", "You'll speak when spoken to, {s_name}",
                                                                                 "Get on your knees", "You're pathetic", "Are you fucking sorry?", "On your knees", "Kiss my ass :kiss:", "Spank Me", f"Spank me, {name}",
                                                                                 "Kiss my feet :kiss:", "You'll need to beg for it", "Bend over", "Suck my cock, slut", "You'd look pretty with my cock in your mouth, don't you think?",
                                                                                 "Suck my dick", "Suck my toes", "Suck my cunt", "Suck my pussy", "Suck my clit", "You're trash", "Don't forget who's in control here", "You're nothing",
                                                                                 "You're mine :smiling_imp:", "Beg for it", "Bend over", f"You've been a naughty {s_name} :smiling_imp:", "You like that?", "Agony awaits :smiling_imp:",
                                                                                 "Who's your Queen? I am.", "Suffer", "You're weak", "Scream for me", "You're nothing but my little fuck toy", "You'd like that, wouldn't you?",
                                                                                 "You deserve this",
                                                                                 "I. Own. You.", f"You deserve this, don't you?", f"Cum for me, {name}", f"Cum for me, my little {s_name}", "This is what you always wanted, isn't it?",
                                                                                 "I own your ass", f"I own your {b_part}", "You're my little slut", "No touching, only watching", f"Lick my ass, {s_name}", "Lick my toes, slut",
                                                                                 "I want you", f"I want {'you inside me' if s_name == 'boy' else 'to be inside you'}", "You'll breathe when I let you", "Please let me cum :pleading_face:",
                                                                                 "I'm yours", "Say my name", "eyy bby, u wan sum fuk?", "You'll cum when I tell you to", "If you want to cum, you'll have to beg for it, slut",
                                                                                 "I'm your robotic fucktoy", "I wanna be ur lil roboslut <:unf:687858959406989318>", "Choke me <:unf:687858959406989318>"
                                                                                                                                                                     f"Choke me, {name}", f"Fuck me, {name}",
                                                                                 " \\*leans in close to your ear\\* You're ||garbage||"]))
    else:
        await message.channel.send(f"{message.author.mention} " + random.choice(["Y-you wanna hold hands? :pleading_face::point_right::point_left:", "Hi, :slight_smile:", ":kiss:", ":slight_smile:", "OwO", "uwu",
                                                                                 "I wanna show you something", ":hibiscus: You're beautiful to me :hibiscus:", "Hello, I love you", "You're perfect", "Never change",
                                                                                 "Hold me :pleading_face:", "I love you so much :smiling_face_with_3_hearts:", "You make me feel special inside",
                                                                                 "You deserve to be loved and respected", "I love and respect you very much", "You are capable of more than you know"]))


def check_haiku(message):
    # count the syllables and check if the message is a haiku
    poem = message.content if hasattr(message, 'content') else message
    parts = [(syllable_count(x),x) for x in re.sub(r'[^\w \n\t]', '', poem).split()]
    # read until we have 5 then 7 then 5
    # pop parts,
    first_line, second_line, third_line = 0, 0, 0
    top_line, middle_line, bottom_line = [], [], []
    try:
        while first_line != 5:
            cnt, wrd = parts.pop(0)
            first_line += cnt
            top_line.append(wrd)
        while second_line != 7:
            cnt, wrd = parts.pop(0)
            second_line += cnt
            middle_line.append(wrd)
        while third_line != 5:
            cnt, wrd = parts.pop(0)
            third_line += cnt
            bottom_line.append(wrd)
        if len(parts) == 0:
            return [' '.join(x) for x in (top_line, middle_line, bottom_line)]
    except:
        return False


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
    if message.author.id == 897853931043037205 and message.channel.id == 570216721512792105:
        await message.delete()
    if message.channel.id == 570213862285115393 and len(message.author.roles) == 1:
        await message.delete()
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
    # urls = []
    # image_types = ("image/jpeg", "image/png",
    #                "video/mp4", "video/webm", "image/gif")
    #for url in URLExtract().find_urls(message.content):
    #    response = requests.head(url)
    #    if response.headers['Content-Type'] in image_types and int(response.headers['Content-Length']) < 10485760:
    #        urls.append(discord.Embed(url=url, type="image"))      
    # if do_nsfw_check and (message.attachments or message.embeds or urls):
    #     images = []  # list of filenames to check
    #     for attachment in message.attachments:
    #         # print("Message has attachment", attachment)
    #         if attachment.content_type in image_types:
    #             # print("Saving attachment", attachment.filename)
    #             with open(attachment.filename, "wb") as temp_file:
    #                 await attachment.save(temp_file)
    #                 images.append(attachment.filename)
    #     for embed in message.embeds + urls:
    #         if embed.type == "image":
    #             fname = urlparse(embed.url).path.split('/')[-1]
    #             print("Fetching " + embed.url + " to " + fname)
    #             with open(fname, 'wb') as out_file:
    #                 out_file.write(requests.get(embed.url).content)
    #             images.append(fname)
    #     for image in images:
    #         # print("Scoring", image)
    #         if not os.path.exists(image):
    #             continue
    #         sfw, nsfw = check_img(nsfw_model, image)
    #         msg = f"{image} scores SFW: {round(100 * sfw, 2)}% NSFW: {round(100 * nsfw, 2)}%"
    #         print(msg)
    #         if "score" in message.content.lower().split():
    #             await message.channel.send(msg)
    #         if nsfw > 0.95:
    #             if message.channel.id != 570213862285115393:  # is_nsfw():
    #                 try:
    #                     await message.channel.send(f"{message.author.mention} Don't post NSFW images outside the nsfw channel. Score was {round(100 * nsfw, 2)}%")
    #                     await message.delete()
    #                 except Exception as e:
    #                     print(f"Couldn't delete image: {e}")
    #             else:
    #                 await message.add_reaction('<:unf:687858959406989318>')
    #         os.unlink(image)
    # check if I've been mentioned
    if 618425432487886879 in message.raw_mentions:
        await handle_bot_mention(message)
        return

    if mcl.strip() == "ff":
        await message.channel.send("**[SM] Friendly fire is disabled.**")
        return
    if _spelling():
        for word in mcl.split():
            if word in _mistakes:
                await message.channel.send(f"{message.author.mention} it's spelt '{_mistakes[word]}'")
    is_haiku = check_haiku(message)
    if is_haiku and not mcl.startswith('.'):
        message.content = f'.poster "{is_haiku[0]}" "{is_haiku[1]}" "{is_haiku[2]}"'
        ctx = await bot.get_context(message)
        await ctx.invoke(bot.get_command('poster'), is_haiku[0], is_haiku[1], is_haiku[2], dall_e=get_dall_ep())
        return
    if not message.author.bot and "poster" in mcl and any(x in mcl for x in ["odds", "chance", "frequency", "how often", "rate"]):
        await message.channel.send(f"The chance of a poster being generated is: {poster_chance*100}% . A haiku will *always* be posterized")
        return
    if gid == ADL_GUILD_ID and np.random.rand() > 1 - poster_chance and not message.content.startswith('.'):
        # split the message evenly into 3
        lines = [" ".join(x) for x in chunkIt(message.content.split(), 3)]
        while len(lines) < 3:
            lines.append("")
        if all(lines):
            message.content = f'.poster "{lines[0]}" "{lines[1]}" "{lines[2]}"'
    await bot.process_commands(message)


if __name__ == "__main__":
    token = Path('tokens')
    while True:
        try:
            bot.run(token.read_text().strip())
        except KeyboardInterrupt:
            sys.exit("Closing")
        except:
            pass
