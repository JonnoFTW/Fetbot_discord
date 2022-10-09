import asyncio
import json
import time
from io import BytesIO
import random
import re

import discord
from discord.ext import commands
from nltk.metrics import distance
import seaborn as sns
import numpy as np
import matplotlib.pylab as plt
import matplotlib.font_manager as fm

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


def iou(a, b):
    aset = set(x for x in a)
    bset = set(x for x in b)
    return len(aset.intersection(bset)) / float(len(aset.union(bset)))


class FetishCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

        async def status_task():
            while True:
                await asyncio.sleep(180)
                await bot.change_presence(activity=discord.Game(name=random.choice(list(fetishes.keys())).title()))

        self.bot.loop.create_task(status_task())

    @commands.command()
    async def table(self, ctx):
        """
        Show a table of kink compatability scores
        """
        sns.set(rc={
            'font.family': ['Noto Sans CJK KR', 'sans-serif', 'Noto Sans', 'Noto Emoji', 'Noto Sans CJK JP', 'sans-serif']
        })
        data = await self.bot.get_db()
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
        arry = np.array(tbl) * 100
        mask = np.zeros_like(arry, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        ax = plt.subplot()
        sns.heatmap(arry, annot=True, linewidth=0.5, cmap="cool",
                    xticklabels=users, yticklabels=users, square=True, mask=mask, ax=ax)
        # for text_obj in ax.get_xticklabels():
        #    text_obj.set_fontname('Uni Sans')
        # for text_obj in ax.get_yticklabels():
        #    text_obj.set_fontname('Uni Sans')
        ax.set_title(
            r"Kink Kompatibility $\left( \frac{A \cap B}{A \cup B} \right)$")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=45, verticalalignment='center')
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

    @commands.command()
    async def add(self, ctx, *args):
        """
        Record that you have a fetish
        """
        data = await self.bot.get_db()
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
        await self.bot.write_db(data)

    @commands.command()
    @commands.is_owner()
    async def add_fetish(self, ctx, *args):
        """
        Add a fetish definition
        """
        f = ' '.join(args)
        if f not in fetishes:
            fetishes[f] = []
            with open('../fetishes.json', 'w') as fh:
                json.dump(fetishes, fh)

            await ctx.send("Done")
        else:
            await ctx.send("Already in there")

    @commands.command()
    async def remove(self, ctx, *args):
        """
        Remove a fetish
        """
        data = await self.bot.get_db()
        fet = get_canonical(' '.join(args))
        if not fet:
            await ctx.send("No such fetish exists")
            return
        aid = str(ctx.author.id)
        if fet in data[aid]:
            data[aid].remove(fet)
            await ctx.send("Done" if random.random() > 0.05 else "Yes, Daddy")
            await self.bot.write_db(data)
        else:
            await ctx.send("You don't have that fetish")

    @commands.command()
    async def find(self, ctx):
        """
        Find your most compatible kinkster
        """
        data = await self.bot.get_db()
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

    @commands.command(name='random')
    async def get_random(self, ctx):
        """
        Tell you a random fetish
        """
        await ctx.send("Why not try " + random.choice(list(fetishes.keys())) + "?")

    @commands.command(name='show_all')
    async def show_all(self, ctx):
        """
        Give you the link to the master list of fetishes
        """
        await ctx.send("All available fetishes are listed here: http://45.248.76.3:5000/fetishes")

    @commands.command(name='list')
    async def list_fets(self, ctx):
        """
        DMs you your fetishes
        """
        data = await self.bot.get_db()
        await ctx.author.send("Your fetishes are " + (', '.join(data.get(str(ctx.author.id), []))))

    @commands.command()
    async def score(self, ctx, *, member: discord.User):
        """
        Get the score for you and someone else
        """
        data = await self.bot.get_db()
        if str(member.id) in data:
            score = iou(data.get(str(ctx.author.id), []),
                        data.get(str(member.id), []))
            await ctx.send(f"{ctx.author.mention} your compatibility with {member.name} is {score}")
        else:
            await ctx.send(f"{ctx.author.mention}: {member.name} is not a kinkster")

    @commands.Cog.listener("on_message")
    async def on_message(self, message):
        if 618425432487886879 not in message.raw_mentions:
            return
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
            await message.channel.send(message.author.mention + " " + random.choice(
                ["You can't talk to me", f"Harder, {name}", "Eyes closed", "You'll speak when spoken to, {s_name}",
                 "Get on your knees", "You're pathetic", "Are you fucking sorry?", "On your knees", "Kiss my ass :kiss:", "Spank Me", f"Spank me, {name}",
                 "Kiss my feet :kiss:", "You'll need to beg for it", "Bend over", "Suck my cock, slut",
                 "You'd look pretty with my cock in your mouth, don't you think?",
                 "Suck my dick", "Suck my toes", "Suck my cunt", "Suck my pussy", "Suck my clit", "You're trash", "Don't forget who's in control here",
                 "You're nothing",
                 "You're mine :smiling_imp:", "Beg for it", "Bend over", f"You've been a naughty {s_name} :smiling_imp:", "You like that?",
                 "Agony awaits :smiling_imp:",
                 "Who's your Queen? I am.", "Suffer", "You're weak", "Scream for me", "You're nothing but my little fuck toy", "You'd like that, wouldn't you?",
                 "You deserve this",
                 "I. Own. You.", f"You deserve this, don't you?", f"Cum for me, {name}", f"Cum for me, my little {s_name}", "This is what you always wanted, isn't it?",
                 "I own your ass", f"I own your {b_part}", "You're my little slut", "No touching, only watching", f"Lick my ass, {s_name}", "Lick my toes, slut",
                 "I want you", f"I want {'you inside me' if s_name == 'boy' else 'to be inside you'}", "You'll breathe when I let you",
                 "Please let me cum :pleading_face:",
                 "I'm yours", "Say my name", "eyy bby, u wan sum fuk?", "You'll cum when I tell you to", "If you want to cum, you'll have to beg for it, slut",
                 "I'm your robotic fucktoy", "I wanna be ur lil roboslut <:unf:687858959406989318>", "Choke me <:unf:687858959406989318>",
                 f"Choke me, {name}", f"Fuck me, {name}",
                 " \\*leans in close to your ear\\* You're ||garbage||"
                 ]))
        else:
            await message.channel.send(f"{message.author.mention} " + random.choice(
                ["Y-you wanna hold hands? :pleading_face::point_right::point_left:", "Hi, :slight_smile:", ":kiss:", ":slight_smile:", "OwO", "uwu",
                 "I wanna show you something", ":hibiscus: You're beautiful to me :hibiscus:", "Hello, I love you", "You're perfect", "Never change",
                 "Hold me :pleading_face:", "I love you so much :smiling_face_with_3_hearts:", "You make me feel special inside",
                 "You deserve to be loved and respected", "I love and respect you very much", "You are capable of more than you know"]))


def setup(ctx):
    ctx.add_cog(FetishCog(ctx))
