import json
import re
import time
import traceback
from glob import glob
from io import BytesIO

import aiohttp
import discord
import ngrok
import numpy as np
from discord.ext import commands
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor

from emoji import get_emoji_regexp

from sklearn.cluster import KMeans

from syllables import syllable_count


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def hex2rgb(s):
    h = s.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


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


def check_haiku(message):
    # count the syllables and check if the message is a haiku
    poem = message.content if hasattr(message, 'content') else message
    parts = [(syllable_count(x), x) for x in re.sub(r'[^\w \n\t]', '', poem).split()]
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


def is_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


class DallError(Exception):
    pass


class ImageGenCog(commands.Cog):
    poster_chance = 0.02

    def __init__(self, bot):
        self.bot = bot

    def get_dall_ep(self):
        client = ngrok.Client(self.bot.key_store['ngrok'])
        for tn in client.tunnels.list():
            if tn.metadata == "dall-e":
                return tn.public_url
        return None

    async def get_dall_e_img(self, ctx, msg, ep):
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

    @commands.command(aliases=["dream", "sd", "diffuse", "diffusion"])
    async def dalle(self, ctx, *, msg):
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
        endpoint = self.get_dall_ep()
        if not endpoint:
            await ctx.send("dream is not running")
            return
        try:
            buff, resp, args, duration, ext = await self.get_dall_e_img(ctx, msg, endpoint)
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

    @commands.command(name='poster')
    async def poster(self, ctx, arg1="", arg2="", arg3="", *args, **kwargs):
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
            is_haiku = kwargs.get('dall_e')
            if is_haiku:
                im_buf, _, _, _, _ = await self.get_dall_e_img(ctx, f'{arg1}, {arg2}, {arg3} --steps 120', self.get_dall_ep())
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
                # draw.rectangle([16-4, y_start-4, ts[0]+16, ts[1]+4], fill=bg_col)
                draw.text((16, y_start), arg1, get_colour_dist(
                    im, y_start, ts), font=fonta, stroke_width=3, stroke_fill="white")
            if arg2:
                fontb, ts = fit_text(
                    arg2, draw, np.random.choice(fonts), basewidth, 96)
                y_start = im.size[1] / 2 - 12
                # draw.rectangle([16-4, y_start-4, ts[0]+4, ts[1]+4], fill=bg_col)
                draw.text((16, y_start), arg2, get_colour_dist(
                    im, y_start, ts), font=fontb, stroke_width=3, stroke_fill="black")
            if arg3:
                fontc, ts = fit_text(
                    arg3, draw, np.random.choice(fonts), basewidth, 76)
                y_start = im.size[1] - 128
                # draw.rectangle([16-4, y_start-4, ts[0]+4, ts[1]+4], fill=bg_col)
                draw.text((16, y_start), arg3, get_colour_dist(
                    im, y_start, ts), font=fontc, stroke_width=3, stroke_fill="black")

            buff = BytesIO()
            im.save(buff, format='png', quality=95)
            buff.seek(0)
            fname = f'poster_{time.time()}.png'
            file = discord.File(buff, filename=fname)
            embed = discord.Embed()
            if is_haiku:
                embed.title = "Haiku"
                embed.description = ctx.message.author.mention
            embed.set_image(url=f'attachment://{fname}')
            await ctx.send(file=file, embed=embed)

    @commands.Cog.listener("on_message")
    async def on_message(self, message: discord.Message):
        mcl = message.content.lower()
        is_haiku = check_haiku(message)
        if is_haiku and not mcl.startswith('.'):
            message.content = f'.poster "{is_haiku[0]}" "{is_haiku[1]}" "{is_haiku[2]}"'
            ctx = await self.bot.get_context(message)
            await self.poster(ctx, is_haiku[0], is_haiku[1], is_haiku[2], dall_e=True)
            return
        if not message.author.bot and "poster" in mcl and any(x in mcl for x in ["odds", "chance", "frequency", "how often", "rate"]):
            await message.channel.send(f"The chance of a poster being generated is: {self.poster_chance * 100}% . A haiku will *always* be posterized")
            return
        if not isinstance(message.channel, discord.DMChannel):
            gid = message.guild.id
            if gid == self.bot.ADL_GUILD_ID and np.random.rand() > 1 - self.poster_chance and not message.content.startswith('.'):
                # split the message evenly into 3
                lines = [" ".join(x) for x in chunkIt(message.content.split(), 3)]
                while len(lines) < 3:
                    lines.append("")
                if all(lines):
                    message.content = f'.poster "{lines[0]}" "{lines[1]}" "{lines[2]}"'
        await self.bot.process_commands(message)


def setup(ctx):
    ctx.add_cog(ImageGenCog(ctx))
