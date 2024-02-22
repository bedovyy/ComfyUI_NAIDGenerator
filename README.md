# ComfyUI_NAIDGenerator
A [ComfyUI](https://github.com/comfyanonymous/ComfyUI) extension for generating image via NovelAI API.

## Installation
- `git clone https://github.com/bedovyy/ComfyUI_NAIDGenerator` into the `custom_nodes` directory.
- or 'Install via Git URL' from [Comfyui Manager](https://github.com/ltdrdata/ComfyUI-Manager)

## Setting up NAI account
Before using the nodes, you should set NAI_ACCESS_TOKEN on `ComfyUI/.env` file.
```
NAI_ACCESS_TOKEN=<ACCESS_TOKEN>
```

You can get persistent API token by **User Settings > Account > Get Persistent API Token** on NovelAI webpage.

Otherwise, you can get access token which is valid for 30 days using [novelai-api](https://github.com/Aedial/novelai-api).

## Usage
The nodes are located at `NovelAI` category.

![image](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/8ab1ecc0-2ba8-4e38-8810-727e50a20923)

### Txt2img
Simply connect `GenerateNAID` node and `SaveImage` node.

![generate](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/1328896d-7d4b-4d47-8ec2-d1c4e8e2561c)

Note that all generated images via `GeneratedNAID` node are saved as `output/NAI_autosave_12345_.png` for keeping original metadata.

### Img2img

Connect `Img2ImgOptionNAID` node to `GenerateNAID` node and put original image.

![image](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/15ff8961-4f6b-4f23-86bf-34b86ace45c0)

Note that width and height of the source image will be resized to generation size.

### Inpainting

Connect `InpaintingOptionNAID` node to `GenerateNAID` node and put original image and mask image.

![image](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/5ed1ad77-b90e-46be-8c37-9a5ee0935a3d)

Note that both source image and mask will be resized fit to generation size.

(You don't need `MaskImageToNAID` node to convert mask image to NAID mask image.)

### Vibe Transfer

Connect `VibeTransferOptionNAID` node to `GenerateNAID` node and put reference image.

![Comfy_workflow](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/8c6c1c2e-f29d-42a1-b615-439155cb3164)

Note that width and height of the source image will be resized to generation size.

### ModelOption

The default model of `GenerateNAID` node is `nai-diffusion-3`(NAI Diffusion Anime V3).

If you want to change model, put `ModelOptionNAID` node to `GenerateNAID` node.

![ModelOption](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/0b484edb-bcb5-428a-b2af-1372a9d7a34f)

### PromptToNAID

ComfyUI use `()` or `(word:weight)` for emphasis, but NovelAI use `{}` and `[]`. This node convert ComfyUI's prompt to NovelAI's.

Optionally, you can choose weight per brace. If you set `weight_per_brace` to 0.10, `(word:1.1)` will convert to `{word}` instead of `{{word}}`.

![image](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/25c48350-7268-4d6f-81fe-9eb080fc6e5a)


