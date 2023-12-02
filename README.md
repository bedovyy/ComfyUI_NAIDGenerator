# ComfyUI_NAIDGenerator
A [ComfyUI](https://github.com/comfyanonymous/ComfyUI) extension for generating image via NovelAI API.

## Installation
- `git clone https://github.com/bedovyy/ComfyUI_NAIDGenerator` into the `custom_nodes` directory.
- or 'Install via Git URL' from [Comfyui Manager](https://github.com/ltdrdata/ComfyUI-Manager)

## Setting up NAI account
Before using the nodes, you should set either NAI_USERNAME/NAI_PASSWORD or NAI_ACCESS_KEY on `ComfyUI/.env` file.

### Setting NAI_ACCESS_KEY (recommended)
Running the below command on `ComfyUI` directory.
(If you use virtual environment such as `venv` or `conda`, you should do **activate** first.)
```
python -c "from custom_nodes.ComfyUI_NAIDGenerator.nodes import get_access_key; print(get_access_key('<YOUR_EMAIL>', '<YOUR_PASSWORD>'))"
```
Then, paste the output on `ComfyUI/.env` file.
```
NAI_ACCESS_KEY=<OUTPUT_TEXT>
```

### Setting NAI_USERNAME and NAI_PASSWORD
Instead of NAI_ACCESS_KEY, you can simply put your email and password of NovelAI account on `ComfyUI/.env` file.
```
NAI_USERNAME=<YOUR_EMAIL>
NAI_PASSWORD=<YOUR_PASSWORD>
```

## Usage
The nodes are located at `NovelAI` category.

![nodes](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/8aeba065-2794-4334-baf9-0f8af90ed895)

### Txt2img
Simply connect `GenerateNAID` node and `SaveImage` node.

![generate](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/1eed3062-2353-4986-8fb0-40f193d014db)

Note that all generated images via `GeneratedNAID` node are saved as `output/NAI_autosave_12345_.png` for keeping original metadata.

### Img2img

Connect `Img2ImgOptionNAID` node to `GenerateNAID` node and put original image.

![Img2ImgOption](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/6e932fb0-bde2-478d-90b6-d14957243bb8)

Note that width and height of the original image should be multiples of 64.

### Inpainting

Connect `InpaintingOptionNAID` node to `GenerateNAID` node and put original image and mask image.

![InpaintingOption](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/bb692c95-51db-43f1-874a-cff3a094c5ea)

Note that,
- width and height of the original image should be multiples of 64
- width and height of mask image should be multiples of 8 and the size should be 1/8 of original image

You can use `ImageToNAIMask` node to convert mask image to NAID mask image.

### ModelOption (Testing)

The default model of `GenerateNAID` node is `nai-diffusion-3`(NAI Diffusion Anime V3).

If you want to change model, put `ModelOptionNAID` node to `GenerateNAID` node.

![ModelOption](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/0b484edb-bcb5-428a-b2af-1372a9d7a34f)
