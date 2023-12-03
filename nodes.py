import random
import dotenv
import base64
from hashlib import blake2b
import argon2

import requests
import json
import time

from os import environ as env
import zipfile
import io
from pathlib import Path
import folder_paths
from datetime import datetime
import torch
import comfy.utils
import math
import numpy as np
from PIL import Image, ImageOps

from .wildcards_nai import NAITextWildcards

# cherry-picked from novelai_api.utils
def argon_hash(email: str, password: str, size: int, domain: str) -> str:
    pre_salt = f"{password[:6]}{email}{domain}"
    blake = blake2b(digest_size=16)
    blake.update(pre_salt.encode())
    salt = blake.digest()
    raw = argon2.low_level.hash_secret_raw(password.encode(), salt, 2, int(2000000 / 1024), 1, size, argon2.low_level.Type.ID,)
    hashed = base64.urlsafe_b64encode(raw).decode()
    return hashed

def get_access_key(email: str, password: str) -> str:
    return argon_hash(email, password, 64, "novelai_data_access_key")[:64]


BASE_URL="https://api.novelai.net"
def login(key) -> str:
    response = requests.post(f"{BASE_URL}/user/login", json={ "key": key })
    response.raise_for_status()
    return response.json()["accessToken"]

def generate_image(access_token, prompt, model, action, parameters, timeout=120.0):
    data = { "input": prompt, "model": model, "action": action, "parameters": parameters }
    response = requests.post(f"{BASE_URL}/ai/generate-image", json=data, headers={ "Authorization": f"Bearer {access_token}" }, timeout=timeout)
    response.raise_for_status()
    return response.content


def imageToBase64(image):
    i = 255. * image[0].cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    image_bytesIO = io.BytesIO()
    img.save(image_bytesIO, format="png")
    return base64.b64encode(image_bytesIO.getvalue()).decode()

def naimaskToBase64(image):
    i = 255. * image[0].cpu().numpy()
    i = np.clip(i, 0, 255).astype(np.uint8)
    alpha = np.sum(i, axis=-1) > 0
    alpha = np.uint8(alpha * 255)
    rgba = np.dstack((i, alpha))
    img = Image.fromarray(rgba)
    image_bytesIO = io.BytesIO()
    img.save(image_bytesIO, format="png")
    return base64.b64encode(image_bytesIO.getvalue()).decode()


class ImageToNAIMask:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "image": ("IMAGE",) } }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "NovelAI/utils"
    def convert(self, image):
        samples = image.movedim(-1,1)
        width = math.ceil(samples.shape[3] / 64) * 8
        height = math.ceil(samples.shape[2] / 64) * 8
        s = comfy.utils.common_upscale(samples, width, height, "nearest-exact", "disabled")
        s = s.movedim(1,-1)
        naimaskToBase64(s)
        return (s,)

class ModelOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["safe-diffusion", "nai-diffusion", "nai-diffusion-furry", "nai-diffusion-2", "nai-diffusion-3"], { "default": "nai-diffusion-3" }),
            },
            "optional": { "option": ("NAID_OPTION",) },
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, model, option=None):
        option = option or {}
        option["model"] = model
        return (option,)

class Img2ImgOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", { "default": 0.70, "min": 0.01, "max": 0.99, "step": 0.01, "display": "number" }),
                "noise": ("FLOAT", { "default": 0.00, "min": 0.00, "max": 0.99, "step": 0.02, "display": "number" }),
            },
#            "optional": { "option": ("NAID_OPTION",) },
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, image, strength, noise, option=None):
        option = option or {}
        option["img2img"] = (image, strength, noise)
        return (option,)

class InpaintingOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "add_original_image": ("BOOLEAN", { "default": True }),
            },
#            "optional": { "option": ("NAID_OPTION",) },
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, image, mask, add_original_image, option=None):
        option = option or {}
        option["infill"] = (image, mask, add_original_image)
        return (option,)


class GenerateNAID:
    def __init__(self):
        dotenv.load_dotenv()
        self.handle_login()
        self.output_dir = folder_paths.get_output_directory()
    
    def handle_login(self):
        """
        Logins and returns an access token
        """
        if "NAI_ACCESS_KEY" in env:
            access_key = env["NAI_ACCESS_KEY"]
        elif "NAI_USERNAME" in env and "NAI_PASSWORD" in env:
            username = env["NAI_USERNAME"]
            password = env["NAI_PASSWORD"]
            access_key = get_access_key(username, password)
        else:
            raise RuntimeError("Please ensure that NAI_ACCESS_KEY or NAI_USERNAME and NAI_PASSWORD are set in your environment")
        access_token = login(access_key)
        self.access_token = access_token
        return access_token

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "size": (["Portrait", "Landscape", "Square", "Random", "Custom(Paid)", "Large Portrait(Paid)", "Large Landscape(Paid)"], { "default": "Portrait" }),
                "width": ("INT", { "default": 832, "min": 64, "max": 1600, "step": 64, "display": "number" }),
                "height": ("INT", { "default": 1216, "min": 64, "max": 1600, "step": 64, "display": "number" }),
                "positive": ("STRING", { "default": "{}, best quality, amazing quality, very aesthetic, absurdres", "multiline": True, "dynamicPrompts": False }),
                "negative": ("STRING", { "default": "lowres", "multiline": True, "dynamicPrompts": False }),
                "steps": ("INT", { "default": 28, "min": 0, "max": 50, "step": 1, "display": "number" }),
                "cfg": ("FLOAT", { "default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1, "display": "number" }),
                "smea": (["none", "SMEA", "SMEA+DYN"], { "default": "none" }),
                "sampler": (["k_euler", "k_euler_ancestral", "k_dpmpp_2s_ancestral", "k_dpmpp_2m", "k_dpmpp_sde", "ddim"], { "default": "k_euler" }),
                "scheduler": (["native", "karras", "exponential", "polyexponential"], { "default": "native" }),
                "seed": ("INT", { "default": 0, "min": 0, "max": 9999999999, "step": 1, "display": "number" }),
                "uncond_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05, "display": "number" }),
                "cfg_rescale": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.02, "display": "number" }),
                "delay_max": ("FLOAT", { "default": 2.1, "min": 2.0, "max": 24000.0, "step": 0.1, "display": "number" }), # delay in seconds
                "fallback_black": ("INT", { "default": 1, "min": 0, "max": 1, "step": 1, "display": "number" }), # 0: no fallback, 1: fallback to black image
            },
            "optional": { "option": ("NAID_OPTION",) },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "NovelAI"
    
    def sanitize_options(self, size, width, height, steps, option):
        """
        Validates the free options
        """
        if "Paid" in size:
            # sanitize Large Portrait and Large Landscape
            if "Large Portrait" in size:
                width = 1024
                height = 1536
            elif "Large Landscape" in size:
                width = 1536
                height = 1024
            return size, width, height, steps, option
        # if width or height is not default, warn and reset to default
        if size == 'Random':
            size = random.choice(["Portrait", "Landscape", "Square"])
        if size == "Portrait":
            if width != 832 or height != 1216:
                print("Overriding width and height to default values for Portrait")
                width = 832
                height = 1216 
        elif size == "Landscape":
            if width != 1216 or height != 832:
                print("Overriding width and height to default values for Landscape")
                width = 1216
                height = 832
        elif size == "Square":
            if width != 1024 or height != 1024:
                print("Overriding width and height to default values for Square")
                width = 1024
                height = 1024
        if steps > 28:
            print("Overriding steps to 28")
            steps = 28
        if option:
            if "img2img" in option or "infill" in option:
                print("Overriding option to None")
                option = {}
        return size, width, height, steps, option

    def generate(self, size, width, height, positive, negative, steps, cfg, smea, sampler, scheduler, seed, uncond_scale, cfg_rescale, delay_max, fallback_black, option=None):
        # ref. novelai_api.ImagePreset
        # We override the default values here for non-custom sizes
        size, width, height, steps, option = self.sanitize_options(size, width, height, steps, option)
        params = {
            "legacy": False,
            "quality_toggle": False,
            "width": width,
            "height": height,
            "n_samples": 1,
            "seed": seed,
            "extra_noise_seed": seed,
            "sampler": sampler,
            "steps": steps,
            "scale": cfg,
            "uncond_scale": uncond_scale,
            "negative_prompt": negative,
            "sm": smea == "SMEA",
            "sm_dyn": smea == "SMEA+DYN",
            "decrisper": False,
            "controlnet_strength": 1.0,
            "add_original_image": False,
            "cfg_rescale": cfg_rescale,
            "noise_schedule": scheduler,
        }

        model = "nai-diffusion-3"
        action = "generate"
        if "Paid" in size:
            if option:
                if "img2img" in option:
                    action = "img2img"
                    image, strength, noise = option["img2img"]
                    params["image"] = imageToBase64(image)
                    params["strength"] = strength
                    params["noise"] = noise
                elif "infill" in option:
                    action = "infill"
                    image, mask, add_original_image = option["infill"]
                    params["image"] = imageToBase64(image)
                    params["mask"] = naimaskToBase64(mask)
                    params["add_original_image"] = add_original_image

                if "model" in option:
                    model = option["model"]

        if action == "infill" and model != "nai-diffusion-2":
            model = f"{model}-inpainting"

        retry_max = 5
        retry_count = 0
        zipped_bytes = None
        while retry_count < retry_max:
            try:
                zipped_bytes = generate_image(self.access_token, positive, model, "generate", params, timeout = delay_max + 30)
                break
            # handle timeout, 500 errors
            except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
                # check 500 Internal Server Error
                if isinstance(e, requests.exceptions.HTTPError):
                    # wait for 60 seconds
                    print(f"retrying {retry_count} after 60 seconds, relogin...")
                    time.sleep(60) # sleep for 60 seconds
                retry_count += 1
                print(f"retrying {retry_count} after 20 seconds, relogin...")
                time.sleep(20) # sleep for 20 seconds
                self.handle_login() # refresh access token
        if zipped_bytes is None and not fallback_black:
            raise RuntimeError("Failed to generate image, possibly due to timeout")
        
        try:
            zipped = zipfile.ZipFile(io.BytesIO(zipped_bytes))
            image_bytes = zipped.read(zipped.infolist()[0]) # only support one n_samples
            i = Image.open(io.BytesIO(image_bytes))
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
        except Exception as exception:
            if fallback_black:
                image = torch.zeros((1, 3, height, width))
            else:
                raise exception
        
        ## save original png to comfy output dir
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path("NAI_autosave", self.output_dir)
        file = f"{filename}_{counter:05}_.png"
        d = Path(full_output_folder)
        d.mkdir(exist_ok=True)
        (d / file).write_bytes(image_bytes)

        if delay_max > 0.0:
            # sleep
            time.sleep(random.randint(2, delay_max))

        return (image,)


NODE_CLASS_MAPPINGS = {
    "GenerateNAID": GenerateNAID,
    "ModelOptionNAID": ModelOption,
    "Img2ImgOptionNAID": Img2ImgOption,
    "InpaintingOptionNAID": InpaintingOption,
    "ImageToNAIMask": ImageToNAIMask,
    "NAITextWildcards": NAITextWildcards # Wildcards
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GenerateNAID": "Generate ✒️🅝🅐🅘",
    "ModelOptionNAID": "ModelOption ✒️🅝🅐🅘",
    "Img2ImgOptionNAID": "Img2ImgOption ✒️🅝🅐🅘",
    "InpaintingOptionNAID": "InpaintingOption ✒️🅝🅐🅘",
    "ImageToNAIMask": "Convert Image to NAI Mask",
    "NAITextWildcards": "NAITextWildcards ✒️🅝🅐🅘"
}