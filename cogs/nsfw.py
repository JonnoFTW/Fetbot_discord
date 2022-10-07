# urls = []
# image_types = ("image/jpeg", "image/png",
#                "video/mp4", "video/webm", "image/gif")
# for url in URLExtract().find_urls(message.content):
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