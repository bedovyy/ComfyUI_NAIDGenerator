import copy
import io
from pathlib import Path
import folder_paths
import zipfile
import json as _json
import copy as _copy

from .utils import *
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import torch
import numpy as np
from PIL import Image as PILImage

TOOLTIP_LIMIT_OPUS_FREE = "Limit image size and steps for free generation by Opus."

# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------

# Accepted canvas sizes (per CR guidance); we will letterbox/pad to one of these
ACCEPTED_CR_SIZES = [(1024, 1536), (1536, 1024), (1472, 1472)]

def _get_user_data(access_token, timeout=120, retry=3):
    """Fetches user data to check Anlas balance. Now a global helper."""
    USER_API_BASE_URL = "https://api.novelai.net"

    req_mod = requests
    if retry is not None and retry > 1:
        retries = Retry(
            total=retry,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retries))
        req_mod = session

    response = req_mod.get(
        f"{USER_API_BASE_URL}/user/data",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=timeout
    )

    response.raise_for_status()
    return response.json()

def _choose_cr_canvas(w, h):
    """Select the accepted CR canvas size whose aspect ratio is closest to the source image."""
    aspect = w / h
    best = None
    best_diff = 9e9
    for cw, ch in ACCEPTED_CR_SIZES:
        diff = abs((cw / ch) - aspect)
        if diff < best_diff:
            best_diff = diff
            best = (cw, ch)
    return best

def pad_image_to_canvas(tensor_image, target_size):
    """
    Letterbox the given tensor image [1,H,W,C] into target_size (W,H) with black padding,
    preserving aspect ratio.
    """
    _, H, W, C = tensor_image.shape
    tw, th = target_size
    arr = (tensor_image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    mode = "RGBA" if (C == 4) else "RGB"
    pil = PILImage.fromarray(arr)

    scale = min(tw / W, th / H)
    new_w = max(1, int(W * scale))
    new_h = max(1, int(H * scale))
    pil_resized = pil.resize((new_w, new_h), PILImage.LANCZOS)

    if mode == "RGBA":
        canvas = PILImage.new("RGBA", (tw, th), (0, 0, 0, 0))
    else:
        canvas = PILImage.new("RGB", (tw, th), (0, 0, 0))
    offset = ((tw - new_w) // 2, (th - new_h) // 2)
    canvas.paste(pil_resized, offset)

    out = np.array(canvas).astype(np.float32) / 255.0
    return torch.from_numpy(out)[None,]

# -------------------------------------------------
# Core simple prompt conversion / utility nodes
# -------------------------------------------------

class PromptToNAID:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
            "text": ("STRING", { "forceInput":True, "multiline": True, "dynamicPrompts": False,}),
            "weight_per_brace": ("FLOAT", { "default": 0.05, "min": 0.05, "max": 0.10, "step": 0.05 }),
            "syntax_mode": (["brace", "numeric"], { "default": "brace" }),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert"
    CATEGORY = "NovelAI/utils"
    def convert(self, text, weight_per_brace, syntax_mode):
        nai_prompt = prompt_to_nai(text, weight_per_brace, syntax_mode)
        return (nai_prompt,)

class ImageToNAIMask:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "image": ("IMAGE",) } }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "NovelAI/utils"
    def convert(self, image):
        s = resize_to_naimask(image)
        return (s,)

class ModelOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ([
                    "nai-diffusion-2",
                    "nai-diffusion-furry-3",
                    "nai-diffusion-3",
                    "nai-diffusion-4-curated-preview",
                    "nai-diffusion-4-full",
                    "nai-diffusion-4-5-curated",
                    "nai-diffusion-4-5-full",
                    "nai-diffusion-5-curated",
                    "nai-diffusion-5-full"
                ], { "default": "nai-diffusion-4-5-full" }),
            },
            "optional": { "option": ("NAID_OPTION",) },
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, model, option=None):
        option = copy.deepcopy(option) if option else {}
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
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, image, strength, noise):
        option = {}
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
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, image, mask, add_original_image):
        option = {}
        option["infill"] = (image, mask, add_original_image)
        return (option,)

class VibeTransferOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "information_extracted": ("FLOAT", { "default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "display": "number" }),
                "strength": ("FLOAT", { "default": 0.6, "min": 0.01, "max": 1.0, "step": 0.01, "display": "number" }),
            },
            "optional": { "option": ("NAID_OPTION",) },
        }
    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, image, information_extracted, strength, option=None):
        option = copy.deepcopy(option) if option else {}
        if "vibe" not in option:
            option["vibe"] = []
        option["vibe"].append((image, information_extracted, strength))
        return (option,)

class NetworkOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ignore_errors": ("BOOLEAN", { "default": True }),
                "timeout_sec": ("INT", { "default": 120, "min": 30, "max": 3000, "step": 1, "display": "number" }),
                "retry": ("INT", { "default": 3, "min": 1, "max": 100, "step": 1, "display": "number" }),
            },
            "optional": { "option": ("NAID_OPTION",) },
        }
    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, ignore_errors, timeout_sec, retry, option=None):
        option = copy.deepcopy(option) if option else {}
        option["ignore_errors"] = ignore_errors
        option["timeout"] = timeout_sec
        option["retry"] = retry
        return (option,)

# -------------------------------------------------
# Character Reference (Single Image)
# -------------------------------------------------

class CharacterReferenceOption:
    INFO_EXTRACT_DEFAULT = 1.0
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "style_aware": ("BOOLEAN", {"default": True, "tooltip": "Copy style along with identity."}),
                "fidelity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number", "tooltip": "How strictly to match the character (and style if enabled)."}),
            },
            "optional": {"option": ("NAID_OPTION",),}
        }
    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, image, style_aware, fidelity, option=None):
        option = copy.deepcopy(option) if option else {}
        fidelity = max(0.0, min(1.0, fidelity))
        option["character_reference_single"] = {
            "image": image,
            "style_aware": style_aware,
            "fidelity": fidelity,
            "info_extracted": self.INFO_EXTRACT_DEFAULT,
        }
        return (option,)

# -------------------------------------------------
# Character Prompt (Multi-Character Text Prompting)
# -------------------------------------------------

class CharacterPromptOption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("STRING", { "multiline": True, "dynamicPrompts": False, "tooltip": "This character's positive caption." }),
                "negative": ("STRING", { "default": "", "multiline": True, "dynamicPrompts": False, "tooltip": "This character's negative caption." }),
                "use_position": ("BOOLEAN", { "default": False, "tooltip": "Pin this character to x/y instead of letting the model place it automatically." }),
                "x": ("FLOAT", { "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number", "tooltip": "Horizontal position (0=left, 1=right). Only used when use_position is enabled." }),
                "y": ("FLOAT", { "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number", "tooltip": "Vertical position (0=top, 1=bottom). Only used when use_position is enabled." }),
                "use_order": ("BOOLEAN", { "default": False, "tooltip": "Use the connection order of character prompt nodes as a priority when characters overlap. Applies to the whole group of chained character prompts." }),
            },
            "optional": { "option": ("NAID_OPTION",) },
        }
    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, positive, negative, use_position, x, y, use_order, option=None):
        option = copy.deepcopy(option) if option else {}
        if "characters" not in option:
            option["characters"] = []
        option["characters"].append({
            "positive": positive, "negative": negative,
            "use_position": use_position, "x": x, "y": y,
            "use_order": use_order,
        })
        return (option,)

# -------------------------------------------------
# Generation Node
# -------------------------------------------------

class GenerateNAID:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "limit_opus_free": ("BOOLEAN", { "default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE }),
                "width": ("INT", { "default": 832, "min": 64, "max": 1600, "step": 64, "display": "number" }),
                "height": ("INT", { "default": 1216, "min": 64, "max": 1600, "step": 64, "display": "number" }),
                "positive": ("STRING", { "default": ", best quality, amazing quality, very aesthetic, absurdres", "multiline": True, "dynamicPrompts": False }),
                "negative": ("STRING", { "default": "lowres", "multiline": True, "dynamicPrompts": False }),
                "steps": ("INT", { "default": 28, "min": 0, "max": 50, "step": 1, "display": "number" }),
                "cfg": ("FLOAT", { "default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1, "display": "number" }),
                "variety" : ("BOOLEAN", { "default": False }),
                "decrisper": ("BOOLEAN", { "default": False }),
                "smea": (["none", "SMEA", "SMEA+DYN"], { "default": "none" }),
                "sampler": (["k_euler", "k_euler_ancestral", "k_dpmpp_2s_ancestral", "k_dpmpp_2m_sde", "k_dpmpp_2m", "k_dpmpp_sde", "ddim"], { "default": "k_euler" }),
                "scheduler": (["native", "karras", "exponential", "polyexponential"], { "default": "native" }),
                "seed": ("INT", { "default": 0, "min": 0, "max": 9999999999, "step": 1, "display": "number" }),
                "uncond_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05, "display": "number" }),
                "cfg_rescale": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.02, "display": "number" }),
                "keep_alpha": ("BOOLEAN", { "default": True, "tooltip": "Disable to further process output images locally" }),
            },
            "optional": { "option": ("NAID_OPTION",) },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "NovelAI"

    @staticmethod
    def _post_image(access_token, prompt, model, action, parameters, timeout=None, retry=None):
        data = {"input": prompt, "model": model, "action": action, "parameters": parameters}

        req_mod = requests
        if retry is not None and retry > 1:
            retries = Retry(total=retry, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST"])
            session = requests.Session()
            session.mount("https://", HTTPAdapter(max_retries=retries))
            req_mod = session

        response = req_mod.post(f"{BASE_URL}/ai/generate-image", json=data, headers={"Authorization": f"Bearer {access_token}"}, timeout=timeout)

        if response.status_code >= 400:
            print("RAW ERROR STATUS:", response.status_code)
            print("RAW ERROR BODY:", response.text)
            try:
                dbg = _copy.deepcopy(data)
                p = dbg.get("parameters", {})
                if "director_reference_images" in p: p["director_reference_images"] = [i[:60] + "...(trunc)" for i in p["director_reference_images"]]
                if "reference_image_multiple" in p: p["reference_image_multiple"] = [i[:60] + "...(trunc)" for i in p["reference_image_multiple"]]
                dbg["parameters"] = p
                print("OUTGOING PAYLOAD (sanitized):", _json.dumps(dbg)[:2000])
            except Exception as e:
                print("Payload debug failed:", e)

        response.raise_for_status()
        return response.content

    def generate(self, limit_opus_free, width, height, positive, negative,
                 steps, cfg, decrisper, variety, smea, sampler, scheduler,
                 seed, uncond_scale, cfg_rescale, keep_alpha, option=None):

        width, height = calculate_resolution(width * height, (width, height))

        params = {
            "params_version": 1, "width": width, "height": height, "scale": cfg, "sampler": sampler, "steps": steps,
            "seed": seed, "n_samples": 1, "ucPreset": 3, "qualityToggle": False,
            "sm": (smea == "SMEA" or smea == "SMEA+DYN") and sampler != "ddim",
            "sm_dyn": (smea == "SMEA+DYN") and sampler != "ddim",
            "dynamic_thresholding": decrisper, "controlnet_strength": 1.0, "legacy": False, "add_original_image": False,
            "cfg_rescale": cfg_rescale, "noise_schedule": scheduler, "legacy_v3_extend": False,
            "uncond_scale": uncond_scale, "negative_prompt": negative, "prompt": positive,
            "reference_image_multiple": [], "reference_information_extracted_multiple": [], "reference_strength_multiple": [],
            "extra_noise_seed": seed,
            "v4_prompt": {"use_coords": False, "use_order": False, "caption": {"base_caption": positive, "char_captions": []}},
            "v4_negative_prompt": {"use_coords": False, "use_order": False, "caption": {"base_caption": negative, "char_captions": []}}
        }

        model = "nai-diffusion-4-5-full"
        action = "generate"

        if sampler == "k_euler_ancestral" and scheduler != "native":
            params["deliberate_euler_ancestral_bug"] = False
            params["prefer_brownian"] = True

        if option:
            if "img2img" in option:
                action = "img2img"
                image, strength, noise = option["img2img"]
                params["image"] = image_to_base64(resize_image(image, (width, height)))
                params["strength"] = strength
                params["noise"] = noise
            elif "infill" in option:
                action = "infill"
                image, mask, add_original_image = option["infill"]
                params["image"] = image_to_base64(resize_image(image, (width, height)))
                params["mask"] = naimask_to_base64(resize_to_naimask(mask, (width, height), "nai-diffusion-4" in model or "nai-diffusion-5" in model))
                params["add_original_image"] = add_original_image

            if "vibe" in option:
                for vibe in option["vibe"]:
                    vimg, information_extracted, strength = vibe
                    params["reference_image_multiple"].append(image_to_base64(resize_image(vimg, (width, height))))
                    params["reference_information_extracted_multiple"].append(information_extracted)
                    params["reference_strength_multiple"].append(strength)

            if "model" in option: model = option["model"]
            if "v4_prompt" in option: params["v4_prompt"].update(option["v4_prompt"])

            if "characters" in option:
                chars = option["characters"]
                use_coords = any(c["use_position"] for c in chars)
                use_order = any(c["use_order"] for c in chars)
                params["v4_prompt"]["caption"]["char_captions"] = [
                    {"char_caption": c["positive"], "centers": [{"x": c["x"], "y": c["y"]}]} for c in chars
                ]
                params["v4_prompt"]["use_coords"] = use_coords
                params["v4_prompt"]["use_order"] = use_order
                params["v4_negative_prompt"]["caption"]["char_captions"] = [
                    {"char_caption": c["negative"], "centers": [{"x": c["x"], "y": c["y"]}]} for c in chars
                ]
                params["v4_negative_prompt"]["use_coords"] = use_coords
                params["v4_negative_prompt"]["use_order"] = use_order

            if "character_reference_single" in option:
                ref = option["character_reference_single"]
                base_caption = "character&style" if ref["style_aware"] else "character"
                ref_img = ref["image"]
                _, h_raw, w_raw, _ = ref_img.shape
                canvas_w, canvas_h = _choose_cr_canvas(w_raw, h_raw)
                padded = pad_image_to_canvas(ref_img, (canvas_w, canvas_h))
                params["director_reference_images"] = [image_to_base64(padded)]
                params["director_reference_descriptions"] = [{"use_coords": False, "use_order": False, "legacy_uc": False, "caption": {"base_caption": base_caption, "char_captions": []}}]
                params["director_reference_strength_values"] = [1.0]
                params["director_reference_secondary_strength_values"] = [1.0 - ref["fidelity"]]
                params["director_reference_information_extracted"] = [1.0]

        timeout = option.get("timeout", 120) if option else 120
        retry = option.get("retry", 3) if option else 3

        if limit_opus_free:
            pixel_limit = 1024 * 1024
            if width * height > pixel_limit:
                params["width"], params["height"] = calculate_resolution(pixel_limit, (width, height))
            if steps > 28: params["steps"] = 28

        if variety: params["skip_cfg_above_sigma"] = calculate_skip_cfg_above_sigma(params["width"], params["height"])
        if sampler == "ddim" and "nai-diffusion-2" not in model: params["sampler"] = "ddim_v3"
        if action == "infill" and "nai-diffusion-2" not in model: model = f"{model}-inpainting"
        
        start_anlas = None
        try:
            user_data = _get_user_data(self.access_token, timeout, retry)
            start_anlas = user_data.get("subscription", {}).get("trainingStepsLeft")
            if start_anlas is not None: print(f"[NovelAI] Anlas (pre-gen): {start_anlas}")
        except Exception as e: print(f"[NovelAI] Anlas tracking failed (pre-gen): {e}")

        image = blank_image()
        try:
            zipped_bytes = self._post_image(self.access_token, positive, model, action, params, timeout, retry)
            with zipfile.ZipFile(io.BytesIO(zipped_bytes)) as zipped:
                image_bytes = zipped.read(zipped.infolist()[0])

            full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path("NAI_autosave", self.output_dir)
            file = f"{filename}_{counter:05}_.png"
            d = Path(full_output_folder)
            d.mkdir(exist_ok=True)
            (d / file).write_bytes(image_bytes)
            
            if start_anlas is not None:
                try:
                    user_data_final = _get_user_data(self.access_token, timeout, retry)
                    final_anlas = user_data_final.get("subscription", {}).get("trainingStepsLeft")
                    if final_anlas is not None:
                        print(f"[NovelAI] Generation cost: {start_anlas - final_anlas} Anlas")
                        print(f"[NovelAI] Anlas (post-gen): {final_anlas}")
                except Exception as e: print(f"[NovelAI] Anlas tracking failed (post-gen): {e}")

            image = bytes_to_image(image_bytes, keep_alpha)
        except Exception as e:
            if option and option.get("ignore_errors", False): print("ignore error:", e)
            else: raise e

        return (image,)

# -------------------------------------------------
# Director Tool Augment Nodes
# -------------------------------------------------

def base_augment(access_token, output_dir, limit_opus_free, ignore_errors, req_type, image, options=None):
    w, h = image.shape[2], image.shape[1]
    if limit_opus_free and w * h > 1024 * 1024:
        w, h = calculate_resolution(1024 * 1024, (w, h))
            
    start_anlas = None
    try:
        user_data = _get_user_data(access_token)
        start_anlas = user_data.get("subscription", {}).get("trainingStepsLeft")
        if start_anlas is not None: print(f"[NovelAI] Anlas (pre-augment): {start_anlas}")
    except Exception as e: print(f"[NovelAI] Anlas tracking failed (pre-augment): {e}")
            
    base64_image = image_to_base64(resize_image(image, (w, h)))
    result_image = blank_image()
    try:
        zipped_bytes = augment_image(access_token, req_type, w, h, base64_image, options=options)
        with zipfile.ZipFile(io.BytesIO(zipped_bytes)) as zipped:
            image_bytes = zipped.read(zipped.infolist()[0])

        full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path("NAI_autosave", output_dir)
        file = f"{filename}_{counter:05}_.png"
        d = Path(full_output_folder)
        d.mkdir(exist_ok=True)
        (d / file).write_bytes(image_bytes)

        if start_anlas is not None:
            try:
                user_data_final = _get_user_data(access_token)
                final_anlas = user_data_final.get("subscription", {}).get("trainingStepsLeft")
                if final_anlas is not None:
                    print(f"[NovelAI] Augment cost: {start_anlas - final_anlas} Anlas")
                    print(f"[NovelAI] Anlas (post-augment): {final_anlas}")
            except Exception as e: print(f"[NovelAI] Anlas tracking failed (post-augment): {e}")

        result_image = bytes_to_image(image_bytes)
    except Exception as e:
        if ignore_errors: print("ignore error:", e)
        else: raise e

    return (result_image,)

class RemoveBGAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "limit_opus_free": ("BOOLEAN", { "default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE }), "ignore_errors": ("BOOLEAN", { "default": False }),}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"
    def augment(self, image, limit_opus_free, ignore_errors):
        return base_augment(self.access_token, self.output_dir, limit_opus_free, ignore_errors, "bg-removal", image)

class LineArtAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "limit_opus_free": ("BOOLEAN", { "default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE }), "ignore_errors": ("BOOLEAN", { "default": False }),}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"
    def augment(self, image, limit_opus_free, ignore_errors):
        return base_augment(self.access_token, self.output_dir, limit_opus_free, ignore_errors, "lineart", image)

class SketchAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "limit_opus_free": ("BOOLEAN", { "default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE }), "ignore_errors": ("BOOLEAN", { "default": False }),}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"
    def augment(self, image, limit_opus_free, ignore_errors):
        return base_augment(self.access_token, self.output_dir, limit_opus_free, ignore_errors, "sketch", image)

class ColorizeAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "limit_opus_free": ("BOOLEAN", { "default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE }), "ignore_errors": ("BOOLEAN", { "default": False }), "defry": ("INT", { "default": 0, "min": 0, "max": 5, "step": 1, "display": "number" }), "prompt": ("STRING", { "default": "", "multiline": True, "dynamicPrompts": False }),}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"
    def augment(self, image, limit_opus_free, ignore_errors, defry, prompt):
        return base_augment(self.access_token, self.output_dir, limit_opus_free, ignore_errors, "colorize", image, options={ "defry": defry, "prompt": prompt })

class EmotionAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()
    strength_list = ["normal", "slightly_weak", "weak", "even_weaker", "very_weak", "weakest"]
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "limit_opus_free": ("BOOLEAN", { "default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE }), "ignore_errors": ("BOOLEAN", { "default": False }), "mood": (["neutral", "happy", "sad", "angry", "scared", "surprised", "tired", "excited", "nervous", "thinking", "confused", "shy", "disgusted", "smug", "bored", "laughing", "irritated", "aroused", "embarrassed", "worried", "love", "determined", "hurt", "playful"], { "default": "neutral" }), "strength": (s.strength_list, { "default": "normal" }), "prompt": ("STRING", { "default": "", "multiline": True, "dynamicPrompts": False }),}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"
    def augment(self, image, limit_opus_free, ignore_errors, mood, strength, prompt):
        prompt = f"{mood};;{prompt}"
        defry = EmotionAugment.strength_list.index(strength)
        return base_augment(self.access_token, self.output_dir, limit_opus_free, ignore_errors, "emotion", image, options={ "defry": defry, "prompt": prompt })

class DeclutterAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "limit_opus_free": ("BOOLEAN", { "default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE }), "ignore_errors": ("BOOLEAN", { "default": False }),}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"
    def augment(self, image, limit_opus_free, ignore_errors):
        return base_augment(self.access_token, self.output_dir, limit_opus_free, ignore_errors, "declutter", image)

# -------------------------------------------------
# Anlas Tracker (Visual Node)
# -------------------------------------------------

class AnlasTrackerNAID:
    def __init__(self):
        self.access_token = get_access_token()
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": { "trigger": ("*",) } # Allows chaining to control execution order
        }

    RETURN_TYPES = ("INT", "STRING",)
    RETURN_NAMES = ("anlas_int", "anlas_string",)
    FUNCTION = "get_anlas"
    CATEGORY = "NovelAI/utils"
    
    def get_anlas(self, trigger=None):
        anlas_count = 0
        try:
            user_data = _get_user_data(self.access_token)
            anlas_count = user_data.get("subscription", {}).get("trainingStepsLeft", 0)
            print(f"[NovelAI] Current Anlas Balance: {anlas_count}")
        except Exception as e:
            print(f"[NovelAI] Failed to fetch Anlas balance: {e}")
            return (0, "Error fetching Anlas")
            
        return (anlas_count, f"{anlas_count} Anlas")

# -------------------------------------------------
# V4 Base / Negative Prompt nodes
# -------------------------------------------------

class V4BasePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"base_caption": ("STRING", { "multiline": True }),}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert"
    CATEGORY = "NovelAI/v4"
    def convert(self, base_caption):
        return (base_caption,)

class V4NegativePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"negative_caption": ("STRING", { "multiline": True }),}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert"
    CATEGORY = "NovelAI/v4"
    def convert(self, negative_caption):
        return (negative_caption,)

# -------------------------------------------------
# Registration
# -------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "GenerateNAID": GenerateNAID,
    "ModelOptionNAID": ModelOption,
    "Img2ImgOptionNAID": Img2ImgOption,
    "InpaintingOptionNAID": InpaintingOption,
    "VibeTransferOptionNAID": VibeTransferOption,
    "NetworkOptionNAID": NetworkOption,
    "CharacterReferenceOptionNAID": CharacterReferenceOption,
    "CharacterPromptOptionNAID": CharacterPromptOption,
    "AnlasTrackerNAID": AnlasTrackerNAID, # New node
    "MaskImageToNAID": ImageToNAIMask,
    "PromptToNAID": PromptToNAID,
    "RemoveBGNAID": RemoveBGAugment,
    "LineArtNAID": LineArtAugment,
    "SketchNAID": SketchAugment,
    "ColorizeNAID": ColorizeAugment,
    "EmotionNAID": EmotionAugment,
    "DeclutterNAID": DeclutterAugment,
    "V4BasePrompt": V4BasePrompt,
    "V4NegativePrompt": V4NegativePrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GenerateNAID": "Generate ✒️🅝🅐🅘",
    "ModelOptionNAID": "ModelOption ✒️🅝🅐🅘",
    "Img2ImgOptionNAID": "Img2ImgOption ✒️🅝🅐🅘",
    "InpaintingOptionNAID": "InpaintingOption ✒️🅝🅐🅘",
    "VibeTransferOptionNAID": "VibeTransferOption ✒️🅝🅐🅘",
    "NetworkOptionNAID": "NetworkOption ✒️🅝🅐🅘",
    "CharacterReferenceOptionNAID": "Character Reference ✒️🅝🅐🅘",
    "CharacterPromptOptionNAID": "Character Prompt ✒️🅝🅐🅘",
    "AnlasTrackerNAID": "Anlas Tracker ✒️🅝🅐🅘", # New node
    "MaskImageToNAID": "Convert Mask Image ✒️🅝🅐🅘",
    "PromptToNAID": "Convert Prompt ✒️🅝🅐🅘",
    "RemoveBGNAID": "Remove BG ✒️🅝🅐🅘",
    "LineArtNAID": "LineArt ✒️🅝🅐🅘",
    "SketchNAID": "Sketch ✒️🅝🅐🅘",
    "ColorizeNAID": "Colorize ✒️🅝🅐🅘",
    "EmotionNAID": "Emotion ✒️🅝🅐🅘",
    "DeclutterNAID": "Declutter ✒️🅝🅐🅘",
    "V4BasePrompt": "V4 Base Prompt ✒️🅝🅐🅘",
    "V4NegativePrompt": "V4 Negative Prompt ✒️🅝🅐🅘",
}
