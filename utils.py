from hashlib import blake2b
import argon2

import base64
import io
import re
import requests
from requests.adapters import HTTPAdapter, Retry
import comfy.utils

import torch
import numpy as np
from PIL import Image, ImageOps

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


def login(key) -> str:
    response = requests.post(f"https://api.novelai.net/user/login", json={ "key": key })
    response.raise_for_status()
    return response.json()["accessToken"]

BASE_URL="https://image.novelai.net"
def generate_image(access_token, prompt, model, action, parameters, timeout=None, retry=None):
    data = { "input": prompt, "model": model, "action": action, "parameters": parameters }

    request = requests
    if retry is not None and retry > 1:
        retries = Retry(total=retry, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST"])
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retries))
        request = session

    response = request.post(f"{BASE_URL}/ai/generate-image", json=data, headers={ "Authorization": f"Bearer {access_token}" }, timeout=timeout)
    response.raise_for_status()
    return response.content


def image_to_base64(image):
    i = 255. * image[0].cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    image_bytesIO = io.BytesIO()
    img.save(image_bytesIO, format="png")
    return base64.b64encode(image_bytesIO.getvalue()).decode()

def naimask_to_base64(image):
    i = 255. * image[0].cpu().numpy()
    i = np.clip(i, 0, 255).astype(np.uint8)
    alpha = np.sum(i, axis=-1) > 0
    alpha = np.uint8(alpha * 255)
    rgba = np.dstack((i, alpha))
    img = Image.fromarray(rgba)
    image_bytesIO = io.BytesIO()
    img.save(image_bytesIO, format="png")
    return base64.b64encode(image_bytesIO.getvalue()).decode()

def bytes_to_image(image_bytes):
    i = Image.open(io.BytesIO(image_bytes))
    i = ImageOps.exif_transpose(i)
    image = np.array(i).astype(np.float32) / 255.0
    return torch.from_numpy(image)[None,]

def blank_image():
    return torch.tensor([[[0]]])

def resize_image(image, size_to):
    samples = image.movedim(-1,1)
    w, h = size_to
    s = comfy.utils.common_upscale(samples, w, h, "bilinear", "disabled")
    s = s.movedim(1,-1)
    return s

def resize_to_naimask(mask, image_size=None):
    samples = mask.movedim(-1,1)
    w, h = (samples.shape[3], samples.shape[2]) if not image_size else image_size
    width = int(np.ceil(w / 64) * 8)
    height = int(np.ceil(h / 64) * 8)
    s = comfy.utils.common_upscale(samples, width, height, "nearest-exact", "disabled")
    s = s.movedim(1,-1)
    return s

def calculate_resolution(pixel_count, aspect_ratio):
    pixel_count = pixel_count / 4096
    w, h = aspect_ratio
    k = (pixel_count * w / h) ** 0.5
    width = int(np.floor(k) * 64)
    height = int(np.floor(k * h / w) * 64)
    return width, height


def prompt_to_stack(sentence):
    result = []
    current_str = ""
    stack = [{ "weight": 1.0, "data": result }]

    for i, c in enumerate(sentence):
        if c in '()':
            # current_str = current_str.strip()
            if c == '(':
                if current_str: stack[-1]["data"].append(current_str)
                stack[-1]["data"].append({ "weight": 1.0, "data": [] });
                stack.append(stack[-1]["data"][-1])
            elif c == ')':
                searched = re.search(r"^(.*):([0-9\.]+)$", current_str)
                current_str, weight = searched.groups() if searched else (current_str, 1.1)
                if current_str: stack[-1]["data"].append(current_str)
                stack[-1]["weight"] = float(weight)
                if stack[-1]["data"] != result:
                    stack.pop()
                else: # no more to pop
                    print("error  :", sentence);
                    print(f"col {i:>3}:", " " * i + "^")
                    # raise Exception('Error durring parsing parentheses', sentence, i, c)
            current_str = ""
        else:
            current_str += c

    if current_str:
        stack[-1]["data"].append(current_str)

    return result

def prompt_stack_to_nai(l, weight_per_brace=0.05):
    result = ""
    for el in l:
        if isinstance(el, dict):
            brace_count = round((el["weight"] - 1.0) / weight_per_brace)
            result += "{" * brace_count + "[" * -brace_count + prompt_stack_to_nai(el["data"]) + "}" * brace_count + "]" * -brace_count
        else:
            result += el
    return result

def prompt_to_nai(prompt, weight_per_brace=0.05):
    return prompt_stack_to_nai(prompt_to_stack(prompt.replace("\(", "（").replace("\)", "）")), weight_per_brace).replace("（", "(").replace("）",")")
