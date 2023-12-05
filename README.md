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

![nodes](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/8aeba065-2794-4334-baf9-0f8af90ed895)

### Txt2img
Simply connect `GenerateNAID` node and `SaveImage` node.

![generate](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/1328896d-7d4b-4d47-8ec2-d1c4e8e2561c)

Note that all generated images via `GeneratedNAID` node are saved as `output/NAI_autosave_12345_.png` for keeping original metadata.

### Img2img

Connect `Img2ImgOptionNAID` node to `GenerateNAID` node and put original image.

![Img2ImgOption](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/be2903a6-abc6-4071-9bb6-b04082282dd3)


Note that width and height of the original image should be multiples of 64.

### Inpainting

Connect `InpaintingOptionNAID` node to `GenerateNAID` node and put original image and mask image.

![InpaintingOption](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/370e9f25-67a7-426d-aa22-cd3f9987f93e)

Note that,
- width and height of the original image should be multiples of 64
- width and height of mask image should be multiples of 8 and the size should be 1/8 of original image

You can use `ImageToNAIMask` node to convert mask image to NAID mask image.

### ModelOption (Testing)

The default model of `GenerateNAID` node is `nai-diffusion-3`(NAI Diffusion Anime V3).

If you want to change model, put `ModelOptionNAID` node to `GenerateNAID` node.

![ModelOption](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/0b484edb-bcb5-428a-b2af-1372a9d7a34f)

