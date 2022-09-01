import argparse, os, sys, glob
from secrets import token_hex
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

import json
from flask import Flask, send_file, request, jsonify, make_response


def save_img_meta(fname, headers):
    meta_fname = 'metadata.json'
    with open(meta_fname, 'r') as f:
        meta = json.load(f)
    meta[fname] = headers
    with open(meta_fname, 'w') as f:
        json.dump(meta, f)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


print("Loading model")
start = time.time()
cfg_path = "configs/stable-diffusion/v1-inference.yaml"

config = OmegaConf.load(cfg_path)
ckpt = "models/ldm/stable-diffusion-v1/sd-v1-4.ckpt"
model = load_model_from_config(config, ckpt)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
print("Model loaded in ", time.time() - start)


def load_img(image):
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def save_grid(all_samples, prompt, n_rows, outpath):
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_rows)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    output_fname = os.path.join(
        outpath, f"{prompt.replace(' ', '-').replace(',', '')[:32]}_{token_hex(8)}.jpg"
    )
    Image.fromarray(grid.astype(np.uint8)).save(output_fname, format="JPEG")
    return output_fname


def img2img(
        prompt,
        input_img,
        ddim_steps=50,
        ddim_eta=0.0,
        scale=7.5,
        n_rows=1,
        strength=0.75,
        seed=False,
        n_iter=1,
        precision='autocast'
):
    if ddim_steps % 2 == 1:
        ddim_steps += 1
    if seed is False:
        seed_everything(int(time.time()))
    else:
        seed_everything(int(seed))
    sampler = DDIMSampler(model)
    outpath = os.path.join("outputs", "webapp")
    os.makedirs(outpath, exist_ok=True)
    n_samples = 1
    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    data = [batch_size * [prompt]]

    init_image = load_img(input_img).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    t_enc = int(strength * ddim_steps)
    precision_scope = autocast if precision == "autocast" else nullcontext

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(n_iter, desc="Sampling", disable=True):
                    for prompts in tqdm(data, desc="data", disable=True):
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))
                        # decode it
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc, )

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        all_samples.append(x_samples)

                # additionally, save as grid
                output_fname = save_grid(all_samples, prompt, n_rows, outpath)
                toc = time.time()
                print(
                    f"Sampling took {toc - tic}s, i.e. produced {n_iter * n_samples / (toc - tic):.2f} samples/sec."
                )
                return output_fname


def generate_from_prompts(
        prompt,
        ddim_steps=50,
        ddim_eta=0.0,
        plms=True,
        fixed_code=False,
        n_iter=1,
        H=512,
        W=512,
        C=4,
        f=8,
        scale=7.5,
        n_rows=1,
        n_samples=1,
        precision="autocast",
        seed=False
):
    """
    prompt=the prompt to render
    ddim_steps="number of ddim sampling steps
    plms=use plms sampling
    fixed_code=if enabled, uses the same starting code across all samples
    ddim_eta=(eta=0.0 corresponds to deterministic sampling)
    n_iter=sample this often
    H=height
    W=width
    C=latent channels
    f=downsampling factor 8 or 16
    scale=unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
    dyn =dynamic thresholding from Imagen, in latent space (TODO: try in pixel space with intermediate decode)
    precision=evaluate at this precision
    """
    if ddim_steps % 2 == 1:
        ddim_steps += 1

    if seed is False:
        seed_everything(int(time.time()))
    else:
        seed_everything(int(seed))
    if plms:
        sampler = PLMSSampler(model, quiet=True)
    else:
        sampler = DDIMSampler(model)

    outpath = os.path.join("outputs", "webapp")
    os.makedirs(outpath, exist_ok=True)

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size

    data = [batch_size * [prompt]]

    start_code = None
    if fixed_code:
        start_code = torch.randn([n_samples, C, H // f, W // f], device=device)

    precision_scope = autocast if precision == "autocast" else nullcontext

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = []
                for n in trange(n_iter, desc="Sampling", disable=True):
                    for prompts in tqdm(data, desc="data", disable=True):
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [C, H // f, W // f]
                        samples_ddim, _ = sampler.sample(
                            S=ddim_steps,
                            conditioning=c,
                            batch_size=n_samples,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=uc,
                            eta=ddim_eta,
                            x_T=start_code,
                        )

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                        )

                        all_samples.append(x_samples_ddim)

                # additionally, save as grid
                output_fname = save_grid(all_samples, prompt, n_rows, outpath)
                toc = time.time()
                print(
                    f"Sampling took {toc - tic}s, i.e. produced {n_iter * n_samples / (toc - tic):.2f} samples/sec."
                )
                return output_fname


app = Flask("stable-diffusion-discord")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000


@app.errorhandler(500)
def error(e):
    return jsonify(error=500, text=str(e)), 500


@app.route("/img2img.jpg", methods=["POST"])
def img_from_img():
    uploaded_file = request.files['file']
    try:
        image = Image.open(uploaded_file.stream).convert("RGB").resize((512,512), Image.Resampling.LANCZOS)
        argfuncmap = dict(
            ddim_steps=int,
            strength=float,
            scale=float,
            seed=int
        )
        args = {
            key: argfuncmap[key](val) for key, val in request.args.items() if key != 'q' and key in argfuncmap
        }
        if 'seed' not in args:
            args['seed'] = int(time.time())
        prompt = request.args['q']
        img_path = img2img(prompt, image, **args)
        response = make_response(send_file(img_path, mimetype="image/jpeg"))
        response.headers['X-SD-Seed'] = args['seed']

        user_headers = {k:v for k,v in request.headers.items() if k.startswith('X-Discord')}
        args['prompt'] = prompt
        out_headers = {
            'user': user_headers,
            'args': args
        }
        save_img_meta(img_path, out_headers)
        return response

    except Exception as e:
        print(e)
        raise Exception("Could not make image: {}".format(e))


@app.route("/img.jpg", methods=["GET"])
def generate():
    prompt = request.args["q"]
    argfuncmap = dict(
        ddim_steps=int,
        H=int,
        W=int,
        scale=float,
        seed=int
    )
    args = {
        key: argfuncmap[key](val) for key, val in request.args.items() if key != 'q' and key in argfuncmap
    }
    if 'seed' not in args:
        args['seed'] = int(time.time())
    print("Rendering prompt", f'"{prompt}" additional args=', args, "user", " ".join(f"{k[10:]}={v}" for k, v in request.headers.items() if k.startswith('X-Discord')))
    try:
        img_path = generate_from_prompts(prompt, **args)
    except Exception as e:
        raise e
    response = make_response(send_file(img_path, mimetype="image/jpeg"))
    response.headers['X-SD-Seed'] = args['seed']
    user_headers = {k:v for k,v in request.headers.items() if k.startswith('X-Discord')}
    args['prompt'] = prompt
    out_headers = {
        'user': user_headers,
        'args': args
    }
    save_img_meta(img_path, out_headers)
    return response
  
