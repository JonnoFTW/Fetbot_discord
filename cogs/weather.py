import json
import pathlib
import time
from datetime import datetime
from ftplib import FTP
from io import BytesIO
from pathlib import Path

import discord
import pandas as pd
import pytz
import requests
from PIL import Image, ImageFont, ImageDraw
from discord.ext import commands
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from nltk.metrics import distance

sites = {}

with (pathlib.Path(__file__).parent.parent / 'bom.dat').open('r') as f:
    for line in f:
        state, ids, location = line.split(' ', 2)
        location = location.title().strip()
        sites[location] = ids.split('.')


def get_suggestions(d, thing):
    return ', '.join([k for k in d.keys() if distance.edit_distance(thing, k) < 5])


def get_guild_city(ctx, key):
    default = {'radar': 'Adelaide', 'weather': 'adelaide'}
    with open('cities.json') as in_file:
        data = json.load(in_file)
        return data.get(str(ctx.guild.id), default)[key]


class WeatherCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def weather(self, ctx, *, site=None):
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
        o = requests.get(url, headers=self.bot.USER_AGENT).json()['observations']['data'][0]
        out = {
            #            'City': o['name'],
            'Temp(°C)': o['air_temp'],
            'Wind(km/h)': o['wind_spd_kmh'],
            'Rain(mm)': o['rain_trace'],
            'Humidity(%)': o['rel_hum'],
            'Wind Dir': o['wind_dir'],
            # 'Visibility(km)': o['vis_km'],
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

    @commands.command()
    async def temps(self, ctx, site=None, field='air_temp', other_field=None):
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

    @commands.command()
    async def bom(self, ctx, *, site=None):
        radars = {
            "Brewarrina": "IDR933",
            "Canberra": "IDR403",
            "Grafton": "IDR283",
            "Hillston": "IDR943",
            "Moree": "IDR533",
            "Namoi": "IDR693",
            "Newcastle": "IDR043",
            "Norfolk Island": "IDR623",
            "Sydney": "IDR713",
            "Wagga Wagga": "IDR553",
            "Wollongong": "IDR033",
            "Yeoval": "IDR963",
            "Alice Springs": "IDR253",
            "Darwin": "IDR633",
            "Gove": "IDR093",
            "Katherine": "IDR423",
            "Warruwi": "IDR773",
            "Bowen": "IDR243",
            "Brisbane": "IDR663",
            "Cairns": "IDR193",
            "Emerald": "IDR723",
            "Gladstone": "IDR233",
            "Greenvale": "IDR743",
            "Gympie": "IDR083",
            "Longreach": "IDR563",
            "Mackay": "IDR223",
            "Marburg": "IDR503",
            "Mornington Island": "IDR363",
            "Mount Isa": "IDR753",
            "Taroom": "IDR983",
            "Townsville": "IDR733",
            "Warrego": "IDR673",
            "Weipa": "IDR783",
            "Willis Island": "IDR413",
            "Adelaide": "IDR643",
            "Adelaide (Sellicks Hill)": "IDR463",
            "Ceduna": "IDR333",
            "Mt Gambier": "IDR143",
            "Woomera": "IDR273",
            "Hobart (Mt Koonya)": "IDR763",
            "Hobart": "IDR373",
            "N.W. Tasmania": "IDR523",
            "Bairnsdale": "IDR683",
            "Melbourne": "IDR023",
            "Mildura": "IDR973",
            "Rainbow": "IDR953",
            "Yarrawonga": "IDR493",
            "Albany": "IDR313",
            "Broome": "IDR173",
            "Carnavon": "IDR053",
            "Dampier": "IDR153",
            "Esperance": "IDR323",
            "Geraldton": "IDR063",
            "Giles": "IDR443",
            "Halls Creek": "IDR393",
            "Kalgoorlie": "IDR483",
            "Learmonth": "IDR293",
            "Newdegate": "IDR383",
            "Perth": "IDR703",
            "Port Hedland": "IDR163",
            "South Doodlakine": "IDR583",
            "Watheroo": "IDR793",
            "Wyndham": "IDR073"
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
        if site is None:
            site = 'Adelaide'
        rid = radars.get(site, radars['Adelaide'])
        prefix = "http://www.bom.gov.au/products/radar_transparencies/"
        fnames = ["IDR.legend.0.png", f"{rid}.background.png",
                  f"{rid}.topography.png", f"{rid}.range.png", f"{rid}.locations.png"]
        out_name = 'radar_animated.gif'
        async with ctx.typing():
            image = None
            for f in fnames:
                p = Path(f)
                if not p.exists():
                    res = requests.get(f"{prefix}{f}", headers=self.bot.USER_AGENT)
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
            if len(images) <= 1:
                await ctx.send(f"{site} might be out of service: http://www.bom.gov.au/products/{rid}.loop.shtml")
                return
            images[0].save(out_name, append_images=images[1:],
                           duration=500, loop=0, save_all=True, optimize=True, disposal=1, include_color_table=True)
            file = discord.File(out_name)
            embed = discord.Embed()
            embed.title = f"BOM Radar for {site}"
            embed.set_image(url=f'attachment://{out_name}')
            await ctx.send(file=file, embed=embed)
            self.cleanup()

    def cleanup(self):
        for p in Path('radar/').glob('*.png'):
            if (time.time() - p.stat().st_ctime) > 3600:
                p.unlink()


def setup(ctx):
    ctx.add_cog(WeatherCog(ctx))
