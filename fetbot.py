#!/usr/bin/env python3
import syllables
from glob import glob
from valve import rcon
import time
from datetime import datetime
import requests
from itertools import cycle
from pathlib import Path
import tabulate
import discord
from discord.ext import commands
import numpy as np
import random
import humanize
import html
import pickle
import sqlite3
import twitter
import json
import asyncio
import re
import cv2
from sklearn.cluster import KMeans
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
import seaborn as sns
import matplotlib.pylab as plt
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from io import BytesIO
from nltk.metrics import distance
import vapeplot
import subprocess
import yaml

import matplotlib.dates as mdates

with open('keys.yaml','r') as infile:
    key_store = yaml.safe_load(infile)
bot = commands.Bot(command_prefix='.')
jokes = json.loads(Path('jokes.json').read_text())
data_store = 'data.json'
lock = asyncio.Lock()

api = twitter.Api(
    key_store['consumer_key'],
    key_store['consumer_secret'],
    key_store['access_token_key'],
    key_store['access_token_secret']
)


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP','Helvetica Neue','Helvetica']

async def get_db():
    async with lock:
        with open(data_store,'r') as fh:
            out = json.load(fh)
            #print(f"Read: {json.dumps(out)}")
            return out

async def write_db(data):
    async with lock:
        with open(data_store,'w') as fh:
            #print(f"Writing: {json.dumps(data)}")
            json.dump(data,fh)

with open('fetishes.json','r') as fh:
    fetishes = json.load(fh)


def strip(msg):
    return re.sub(r'\s+', ' ',re.sub(r"[;-_,]",'',msg.lower()))

def get_canonical(msg):
    s =  msg
    if s in fetishes:
        return s
    for k,v in fetishes.items():
        if k == s or s in v:
            return k

def find_similar(fet):
    thresh = 3
    out = []
    for f,v in fetishes.items():
        if any(distance.edit_distance(fet,x) <= thresh for x in [f]+v):
            out.append(f)
    return ', '.join(out)

async def status_task():
    while True:
        await asyncio.sleep(180)
        await bot.change_presence(activity=discord.Game(name=random.choice(list(fetishes.keys())).title()))

@bot.check
async def restrict_to(ctx):
    return ctx.message.channel.id in {570213862285115393,570216721512792105,570248053567782922,648525630857674762} or not hasattr(ctx.message,'server')

@bot.event
async def on_ready():
    print(f"Logged on")
    await bot.change_presence(activity=discord.Game(name="Cock/Ball Torture"))
    bot.loop.create_task(status_task())

@bot.command()
@commands.is_owner()
async def add_fetish(ctx, *args):
    """
    Add a fetish definition
    """
    f = ' '.join(args)
    if f not in fetishes:
        fetishes[f] = []
        with open('fetishes.json','w') as fh:
            json.dump(fetishes, fh)

        await ctx.send("Done")
    else:
        await ctx.send("Already in there")

@bot.command(prefix='')
async def asl(ctx):
    """a/s/l"""
    places = ['sa', 'hawaii', 'israel', 'nigeria', 'aus', 'cali', 'nyc', 'nsw', 'fl', 'uk', 'france', 'russia', 'germany', 'japan', 'china', 'nz', 'uganda']
    await ctx.send('/'.join([str(np.random.randint(8, 30)), random.choice(['m', 'f']), random.choice(places)]))

@bot.command()
async def flip(ctx):
    """Flip a coin"""
    await ctx.send(f"A coin is flipped: {random.choice(['head','tail'])}")

@bot.command()
async def roulette(ctx, user_a: discord.User, user_b: discord.User):
    it = cycle([user_b, user_a])
    c_user = next(it)
    msg = f"Comrades {user_a.display_name} and {user_b.display_name} have volunteered to play Russian Roulette"
    msg += f"\nA single round is loaded into the revolver and placed against {c_user.display_name}'s temple 😨🔫"
    await ctx.send(msg)
    pos = np.random.randint(0,6)
    current_pos = 0
    while True:
        time.sleep(1)
        msg = "I pull the trigger..."
        if pos == current_pos:
            await ctx.send(msg+f" **BANG** . {c_user.display_name}'s body slumps to the floor as blood splatters your friends")
            break
        else:
            await ctx.send(msg+f" **click**. The revolver is placed against {c_user.display_name}'s head")
        c_user = next(it)
        current_pos += 1


@bot.command()
async def fortune(ctx):
    """
    Give you a fortune
    """
    msg = subprocess.check_output(["fortune","-s"]).decode('utf-8').strip()
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
    Add a fetish
    """
    data = await get_db()
    stripped = strip(' '.join(args))
    fet = get_canonical(stripped)
    if not fet:
        s = "No such fetish exists"
        suggestions = find_similar(stripped)
        if suggestions:
            s += ". Perhaps you meant: "+suggestions
        await ctx.send(s)
        return
    data[str(ctx.author.id)] = list(set(data.get(str(ctx.author.id),[]) + [fet]))
    print(data)
    await ctx.send("Done" if random.random() > 0.05 else "Yes, Daddy")
    await write_db(data)

@bot.command()
@commands.is_owner()
async def pool(ctx, user_a:discord.User, user_b: discord.User):
    data = await get_db()
    if 'pool' not in data:
        data['pool'] = []
    data['pool'].append([user_a.id, user_b.id])
    await write_db(data)
    await ctx.send(f"{user_a.name} beat {user_b.name} in CBT")

@bot.command()
@commands.is_owner()
async def set_kink(ctx):
    
    msg = await ctx.bot.get_channel(570223709022060547).fetch_message(570432571591360532)
    await msg.add_reaction('👺')
  #  for reaction in msg.reactions:
  #      users = await reaction.users().flatten()
  #      for i in users:
  #          print(f"{reaction.emoji}=>{i}")
@bot.command()
async def leaderboard(ctx):
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
    Show a table of scores
    """
    data = await get_db()
    tbl = []
    users = []
    for x in data.keys():
        row = []
        uname = ctx.bot.get_user(int(x))

        users.append(uname.name if uname is not None else "?")
        for y in data.keys():
            row.append(iou(data[x],data[y]))
        tbl.append(row)
    #oldcmap = 'YlGnBu'
    arry = np.array(tbl)
    mask = np.zeros_like(arry, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    ax = plt.subplot()
    sns.heatmap(arry, annot=True, linewidth=0.5, cmap="cool",xticklabels=users,yticklabels=users,square=True,mask=mask, ax=ax)
    ax.set_title(r"Kink Kompatibility $\left( \frac{A \cap B}{A \cup B} \right)$")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, verticalalignment='center')
    buff = BytesIO()
    plt.savefig(buff,format='png')
    buff.seek(0)
    plt.cla()
    plt.clf()
    #plt.savefig("table.png")
    fname = f'table_{time.time()}.png'
    file=  discord.File(buff, filename=fname)
    embed = discord.Embed()
    embed.set_image(url=f'attachment://{fname}')
    await ctx.send(file=file,embed=embed)

@bot.command()
async def bom(ctx):
    now = datetime.now()
    data = requests.get("http://www.bom.gov.au/radar/IDR643.gif").content
    buff = BytesIO(data)
    file = discord.File(buff, filename="radar.gif")
    embed = discord.Embed()
    embed.set_image(url=f'attachment://radar.gif')
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

def iou(a,b):
    aset = set(a)
    bset = set(b)
    return len(aset.intersection(bset))/float(len(aset.union(bset)))

@bot.command()
async def find(ctx):
    """
    Find your most compatible kinkster
    """
    data = await get_db()
    if len(data) <= 1:
        await ctx.send("There's nobody else to compare yourself to")
        return
    score = max(
                sorted([
                    (
                        iou(data.get(str(ctx.author.id),tuple()), data.get(str(other),tuple())),other
                    ) for other in data.keys() if other != str(ctx.author.id)
                ], reverse=True)
            )
    await ctx.send(f"{ctx.author.mention} your best match is <@{score[1]}> with score {round(score[0],3)}")

@bot.command(name='random')
async def get_random(ctx):
    """
    Tell you a random fetish
    """
    await ctx.send("Why not try "+random.choice(list(fetishes.keys()))+"?")

@bot.command(name='show_all')
async def show_all(ctx):
    """
    Give you the link to the master list of fetishes
    """
    await ctx.send("All available fetishes are listed here: https://gist.github.com/JonnoFTW/7788169e843a685e37628d9cd8a0be6b")
@bot.command()
async def weather(ctx):
    """
    Weather details for adelaide
    """
    url = "http://reg.bom.gov.au/fwo/IDS60901/IDS60901.94648.json"
    o = requests.get(url).json()['observations']['data'][0]
    out = {
#            'City': o['name'],
            'Temp(°C)': o['air_temp'],
            'Wind(km/h)': o['wind_spd_kmh'],
            'Rain(mm)':o['rain_trace'],
            'Humidity(%)': o['rel_hum'],
            'Wind Dir': o['wind_dir'],
            'Visibility(km)': o['vis_km'],
            'Updated':o['local_date_time']
        }
#    await ctx.send("```"+tabulate.tabulate(out, headers='keys', tablefmt='plain')+"```")

    embed = discord.Embed(title=o['name'], colour=0x006064, url="http://www.bom.gov.au/products/IDS60901/IDS60901.94648.shtml")
    embed.set_thumbnail(url=f"http://www.bom.gov.au/radar/IDR643.gif?t={time.time()}")
    for k,v in out.items():
        embed.add_field(name=f"**{k}**", value=f"\n{v}")
    await ctx.send(embed=embed)
@bot.command()
async def temps(ctx):
  """
  Shows a chart of the recent temperatures in Adelaide
  """
  async with ctx.typing():
    url = "http://reg.bom.gov.au/fwo/IDS60901/IDS60901.94648.json"
    data = requests.get(url).json()['observations']['data']
    arry = np.array([[datetime.strptime(o['local_date_time_full'],"%Y%m%d%H%M%S"), o['air_temp']] for o in data[-1::-1]])

    x = arry[:,0]
    y = arry[:,1]
    fig, ax = plt.subplots()
    plt.grid(which='both')
    plt.ylabel("Temperature °C")
    plt.title("Temperature In Adelaide")
    ax.plot(x,y)
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%I %p"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    for text in ax.get_xminorticklabels():
        text.set_rotation(70)
        text.set_fontsize('x-small')
    ax.xaxis.set_tick_params(rotation=70)
    buff = BytesIO()
    plt.savefig(buff,format='png')
    buff.seek(0)
    plt.cla()
    plt.clf()
    fname = f'temps_{time.time()}.png'
    file=  discord.File(buff, filename=fname)
    embed = discord.Embed()
    embed.set_image(url=f'attachment://{fname}')
    await ctx.send(file=file,embed=embed)


from emoji import get_emoji_regexp, emojize
emo_re = get_emoji_regexp()
def fit_text(txt,draw, font_path, imwidth, size=128, padding=32):
    print("txt has emoji: ",emo_re.search(txt), txt)
    while 1:
        if emo_re.search(txt):
            try:
#                imfont = ImageFont.truetype("/home/jonno/.fonts/Symbola.ttf",size=size, encoding='unic')
                imfont = ImageFont.truetype("/home/jonno/.fonts/NotoEmoji-Regular.ttf",size=size, encoding='unic')
            except OSError:
                size = size - 2
                continue
        else:
            imfont = ImageFont.truetype(font_path, size=size)
        tw = draw.textsize(txt, imfont)
        if tw[0] < imwidth -32 or size < 16:
            return imfont, tw
        else:
            size = size - 2

with open('colours.json','r') as infile:
    colours = json.load(infile)
all_colours = []
for br in colours.values():
    for col_list in br.values():
        all_colours.extend([sRGBColor.new_from_rgb_hex(c) for c in col_list])

def hex2rgb(s):
    h = s.lstrip('#')
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

def get_colour(im, y_start):
    region = im.crop((0, y_start, im.size[0], y_start + 64)).filter(ImageFilter.BLUR)
    arr = np.array(region)
    avg = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    # bright background, use dark colour font
    use_dark = (avg * np.array([0.299,0.587,0.114])).sum() > 186
    font_br = "light" if use_dark else "dark"
    key = np.random.choice([k for k in colours.keys() if font_br in colours[k]])
    cols = colours[key][font_br]
    print("Use dark=", use_dark)
    return hex2rgb(np.random.choice(cols))

def get_colour_dist(im, y_start ,ts, x_start=16):
    region = np.array(im.crop((x_start, y_start, x_start+ts[0], y_start + ts[1])))
    # take histogram of region , find colour furthest from the most common
    region = region.reshape((region.shape[0]*region.shape[1], 3))
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(region)
    dc_rgb = kmeans.cluster_centers_[0]
    dc_lab = convert_color(sRGBColor(*dc_rgb, is_upscaled=True), LabColor)
    # for each color in all_colors, get the max, might be made parallel
    res = max(all_colours, key=lambda a: delta_e_cie2000(dc_lab, convert_color(a,LabColor)))
    return res.get_upscaled_value_tuple()


@bot.command(name='poster')
#@commands.has_role("Admin")
async def poster(ctx, arg1="", arg2="", arg3=""):
    """
    Generate a cool posters. Use quote marks to separate lines eg: .poster "first line" "" "third line"
    """
    print(f"poster args= '{arg1}' '{arg2}' '{arg3}' ")
    async with ctx.typing():
        arg1 = await commands.clean_content().convert(ctx, arg1)
        arg2 = await commands.clean_content().convert(ctx, arg2)
        arg3 = await commands.clean_content().convert(ctx, arg3)
        im = Image.open(BytesIO(requests.get("https://source.unsplash.com/random").content))
        basewidth = 960
        wpercent = (basewidth / float(im.size[0]))
        hsize =  int((float(im.size[1])*float(wpercent)))
        im = im.resize((basewidth, hsize))
        font_dir = "/home/jonno/.fonts/*.ttf"
        fonts = [f for f in glob(font_dir) if re.search(r"Noto(Sans|Serif|SerifDisplay)-", f)]
        draw = ImageDraw.Draw(im)
        if arg1:
            # replace any mentions with the server name
            fonta, ts = fit_text(arg1, draw, np.random.choice(fonts), basewidth, 106)
            y_start = 16
            draw.text((16, y_start), arg1, get_colour_dist(im, y_start, ts), font=fonta)
        if arg2:
            fontb, ts = fit_text(arg2, draw, np.random.choice(fonts), basewidth, 96)
            y_start = im.size[1]/2 - 12
            draw.text((16, y_start), arg2, get_colour_dist(im, y_start, ts) ,font=fontb)
        if arg3:
            fontc, ts = fit_text(arg3, draw, np.random.choice(fonts), basewidth, 76)
            y_start = im.size[1] - 128
            draw.text((16, y_start), arg3, get_colour_dist(im, y_start, ts), font=fontc)

        buff = BytesIO()
        im.save(buff,format='png', quality=95)
        buff.seek(0)
        fname = f'poster_{time.time()}.png'
        file=  discord.File(buff, filename=fname)
        embed = discord.Embed()
        embed.set_image(url=f'attachment://{fname}')
        await ctx.send(file=file,embed=embed)

@bot.command()
async def set_lastfm(ctx, username):
    db = await get_db()
    if 'lfm' not in db:
        db['lfm'] = {}
    db['lfm'][str(ctx.author.id)] = username
    await write_db(db)
    await ctx.send(f"Set {ctx.author.name}'s lastfm username to '{username}'")
@bot.command(name='np')
async def now_playing(ctx, user=''):
    key = key_store['last_fm_api']
    async with ctx.typing():
        if not user:
            db = await get_db()
            user = db.get('lfm',{}).get(str(ctx.author.id), None)
            if user is None:
                await ctx.send(f"No lastfm name set, use .set_lastfm <username>")
                return
        print("Using username", user)
        data = requests.get("http://ws.audioscrobbler.com/2.0/", {'method':'user.getrecenttracks','user':user,'format':'json','api_key':key}).json()
        song = data['recenttracks']['track'][0]
        msg = f"**{user}** {'is now listening' if '@attr' in song else 'last listened'} to \"*{song['name']}*\" by {song['artist']['#text']} from *{song['album']['#text']}*\n{song['url']}"

        try:
            im_data = requests.get(song['image'][2]['#text']).content
            buff = BytesIO(im_data)
            file = discord.File(buff, filename="cover.jpg")
            embed = discord.Embed()
            embed.set_image(url=f'attachment://cover.jpg')
            await ctx.send(msg, file=file, embed=embed)
        except Exception as e:
            print(e)
            await ctx.send(msg)

@bot.command()
async def genre(ctx):
    prefixes = ['enterprise','', 'post', 'indie', 'avant-garde', 'nautical', 'break', 'wub', 'chip', 'vintage', 'classic', 'virtuosic', 'death', 'instrumental', 'british', 'industrial', 'thrash', 'japanese', 'J', 'K', 'acoustic', 'progressive', 'power', 'glam', 'melodic', 'new wave', 'german', 'gothic', 'symphonic', 'grind', 'synth', 'minimal', 'psychedelic', 'brutal', 'sexy', 'easy listening', 'christian', 'anime', 'stoner', 'comedy', 'sad', 'christmas', 'neo', 'russian', 'finnish', 'summer', 'underground', 'dream', 'pagan', 'minimal', 'ambient', 'nu', 'speed', 'contemporary', 'alt', 'acid', 'english', 'kvlt', 'cult', 'mu', 'raw', 'norwegian', 'viking', 'porn']
    suffixes = ['core', '', 'step', 'groove', 'noise']
    gens = ['folk', 'ambient', 'electronica', 'funk', 'hip-hop', 'dance', 'pop', 'trance', 'indie', 'soul', 'hard', 'lounge', 'blues', 'classical', 'grunge', '/mu/core', 'emo', 'rap', 'rock', 'punk', 'alternative', 'nautical', 'electro', 'swing', 'screamo', 'jazz', 'reggae', 'metal', 'classical', 'math', 'nerd', 'country', 'western', 'dub', "drum 'n' bass", 'celtic', 'shoegaze']
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
    await ctx.author.send("Your fetishes are "+(', '.join(data.get(str(ctx.author.id),[]))))

@bot.command()
async def score(ctx, *, member: discord.User):
    """
    Get the score for you and someone else
    """
    data = await get_db()
    if str(member.id) in data:
        score = iou(data.get(str(ctx.author.id),[]), data.get(str(member.id),[]))
        await ctx.send(f"{ctx.author.mention} your compatibility with {member.name} is {score}")
    else:
        await ctx.send(f"{ctx.author.mention}: {member.name} is not a kinkster")

@bot.command()
async def nextmap(ctx):
    """
    Shows the nextmap at bhop.rip
    """
    nm  = do_rcon('nextmap')
    await ctx.send(nm)

@bot.command()
async def players(ctx):
    """
    Shows the current players of bhop.rip
    """
    players = re.findall(r'^players\s+:\s(.*)$', do_rcon('status'), re.M)[0]
    await ctx.send("**[SM]** Players: "+players)

@bot.command()
async def bhop(ctx):
    """
    Shows server details of bhop.rip
    """
    players = re.findall(r'^hostname:\s(.*)$', do_rcon('status'), re.M)[0]
    await ctx.send("**[SM]** "+players)

@bot.command()
async def currentmap(ctx):
    """
    Shows the current map at bhop.rip
    """
    map = re.findall(r'^map\s+:\s(\w+) ',do_rcon('status'), re.M)[0]
    await ctx.send("**[SM]** Current map is: "+map)
@bot.command()
async def thetime(ctx):
    await ctx.send(f"The time is: {datetime.now()}")
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
    return rcon.execute(addr,pw,cmd)

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

@bot.command(name='twitter')
async def get_twitter(ctx, user):
    """
    Show the last tweet of a given twitter user
    """
    try:
        t = api.GetUserTimeline(screen_name=user)[0]
        await ctx.send("@%s %s: %s" % (t.user.screen_name, humanize.naturaltime(datetime.utcnow() - datetime.strptime(t.created_at,'%a %b %d %H:%M:%S +0000 %Y')), "https://twitter.com/{}/status/{}".format(t.user.screen_name,t.id_str)))
    except Exception as e:
        await ctx.send(f"Error Getting twitter: {e}")

async def handle_bot_mention(message):
    is_dm = type(message.channel) is discord.DMChannel
    if is_dm or message.channel.is_nsfw():
        if is_dm:
            is_female = random.choice([True,False])
        else:
            is_female = 570225951657689088 in [r.id for r in message.author.roles]
        if is_female:
            name = "Mommy"
            s_name = "girl"
            b_part = "cunt"
        else:
            name = "Daddy"
            s_name = "boy"
            b_part = "cock"
        await message.channel.send(message.author.mention +" "+ random.choice(["You can't talk to me", f"Harder, {name}", "Eyes closed", "You'll speak when spoken to, {s_name}",
                                                  "Get on your knees", "You're pathetic", "Are you fucking sorry?", "On your knees", "Kiss my ass :kiss:", "Spank Me", f"Spank me, {name}",
                                                  "Kiss my feet :kiss:", "You'll need to beg for it", "Bend over", "Suck my cock, slut", "You'd look pretty with my cock in your mouth, don't you think?",
                                                  "Suck my dick", "Suck my toes", "Suck my cunt", "Suck my pussy", "Suck my clit", "You're trash", "Don't forget who's in control here", "You're nothing",
                                                  "You're mine :smiling_imp:", "Beg for it", "Bend over", f"You've been a naughty {s_name} :smiling_imp:", "You like that?", "Agony awaits :smiling_imp:",
                                                  "Who's your Queen? I am.", "Suffer", "You're weak", "Scream for me", "You're nothing but my little fuck toy", "You'd like that, wouldn't you?", "You deserve this",
                                                  "I. Own. You.", f"You deserve this, don't you?", f"Cum for me, {name}", f"Cum for me, my little {s_name}", "This is what you always wanted, isn't it?",
                                                  "I own your ass", f"I own your {b_part}", "You're my little slut", "No touching, only watching", f"Lick my ass, {s_name}", "Lick my toes, slut",
                                                  "I want you", f"I want {'you inside me' if s_name=='boy' else 'to be inside you'}", "You'll breathe when I let you", "Please let me cum :pleading_face:",
                                                  "I'm yours", "Say my name", "eyy bby, u wan sum fuk?", "You'll cum when I tell you to", "If you want to cum, you'll have to beg for it, slut",
                                                  "I'm your robotic fucktoy", "I wanna be ur lil roboslut <:unf:687858959406989318>", "Choke me <:unf:687858959406989318>"
                                                  f"Choke me, {name}", f"Fuck me, {name}"," \\*leans in close to your ear\\* You're ||garbage||"]))
    else:
        await message.channel.send(f"{message.author.mention} "+ random.choice(["Y-you wanna hold hands? :pleading_face::point_right::point_left:", "Hi, :slight_smile:", ":kiss:", ":slight_smile:", "OwO","uwu",
                                                                                 "I wanna show you something", ":hibiscus: You're beautiful to me :hibiscus:", "Hello, I love you", "You're perfect", "Never change",
                                                                                 "Hold me :pleading_face:", "I love you so much :smiling_face_with_3_hearts:", "You make me feel special inside",
                                                                                 "You deserve to be loved and respected", "I love and respect you very much", "You are capable of more than you know"]))

@bot.event
async def on_message(message):
    if type(message.channel) is discord.DMChannel:
        name = "DirectMessage"
    else:
        name = message.channel.name
    print(f"#{name}\t{message.author.name}: {message.content} (mentions: {message.raw_mentions})")
    if "mynudesforfree" in re.sub(r'[\W_]', '', message.content):
        print(f"Kicking {message.author.name} for saying: '{message.content}'")
        await message.author.kick(reason="TRY GETTING A RESERVATION AT DORSIA NOW YOU STUPID FUCKING BASTARD")
        await message.delete()

    # check if I've been mentioned
    if 618425432487886879 in message.raw_mentions:
        await handle_bot_mention(message)
        return
    if  message.content.lower().strip() == "ff":
        await message.channel.send("**[SM] Friendly fire is disabled.**")
        return
    """
    # count the syllables and check if the message  is a haiku
    parts = [(syllables.estimate(x),x) for x in message.content.split()]
    # read until we have 5 then 7 then 5
    is_haiku = False
    lines = []
    s,total = [],0
    for word in parts:
        total += word[0]
        s.append(word[1])
        if (len(lines) in [0,2] and total == 5) or (len(lines) == 1 and total == 7):
            lines.append(" ".join(s))
            total = 0
    if len(lines) == 3 and total == 0:
        message.content = f'.poster "{lines[0]}" "{lines[1]}" "{lines[2]}"'
    """
    if np.random.rand() > 0.98:
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
            exit("Closing")
        except:
            pass
