import random
from io import BytesIO

import discord
import requests
from discord.ext import commands


class LastFmCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def set_lastfm(self, ctx, username):
        """
        Let me know your last.fm username
        """
        db = await self.bot.get_db()
        if 'lfm' not in db:
            db['lfm'] = {}
        db['lfm'][str(ctx.author.id)] = username
        await self.bot.write_db(db)
        await ctx.send(f"Set {ctx.author.name}'s lastfm username to '{username}'")

    @commands.command(name='np')
    async def now_playing(self, ctx, user=''):
        """
        Fetch the song you are currently playing from last.fm
        """
        key = self.bot.key_store['last_fm_api']
        async with ctx.typing():
            if not user:
                db = await self.bot.get_db()
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
                    'artist': song['artist']['#text'], 'track': song['name']}).json()
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

    @commands.command()
    async def genre(self, ctx):
        """Come up with your new favourite genre"""
        prefixes = ['enterprise', '', 'post', 'indie', 'avant-garde', 'nautical', 'break', 'wub', 'chip', 'vintage', 'classic', 'virtuosic', 'death', 'instrumental', 'british', 'industrial', 'thrash', 'japanese', 'J', 'K', 'acoustic',
                    'progressive', 'power', 'glam', 'melodic', 'new wave', 'german', 'gothic', 'symphonic', 'grind', 'synth',
                    'minimal', 'psychedelic', 'brutal', 'sexy', 'easy listening', 'christian', 'anime', 'stoner', 'comedy', 'sad', 'christmas', 'neo', 'russian', 'finnish', 'summer', 'underground', 'dream', 'pagan', 'minimal', 'ambient',
                    'nu',
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


def setup(ctx):
    ctx.add_cog(LastFmCog(ctx))
