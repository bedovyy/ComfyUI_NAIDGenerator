# ComfyUI_NAIDGenerator

A [ComfyUI](https://github.com/comfyanonymous/ComfyUI) extension for generating images via the NovelAI API.

## Installation

- `git clone https://github.com/bedovyy/ComfyUI_NAIDGenerator` into the `custom_nodes` directory.
- or 'Install via Git URL' from [Comfyui Manager](https://github.com/ltdrdata/ComfyUI-Manager)

## Setting up NAI account

Before using the nodes, you should set `NAI_ACCESS_TOKEN` in a `.env` file located in your main `ComfyUI` directory.

`ComfyUI/.env`
```
NAI_ACCESS_TOKEN=<YOUR_ACCESS_TOKEN>
```

You can get a persistent API token by navigating to **User Settings > Account > Get Persistent API Token** on the NovelAI website.

Otherwise, you can get an access token which is valid for 30 days using [novelai-api](https://github.com/Aedial/novelai-api).

## Usage

The nodes are located in the `NovelAI` category.

![image](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/8ab1ecc0-2ba8-4e38-8810-727e50a20923)

### Txt2img

Simply connect the `GenerateNAID` node to a `SaveImage` node.

![generate](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/1328896d-7d4b-4d47-8ec2-d1c4e8e2561c)

**Note:** All generated images via the `GenerateNAID` node are automatically saved to `output/NAI_autosave/NAI_autosave_#####_.png` to preserve their original metadata.

### Img2img

Connect an `Img2ImgOptionNAID` node to the `option` input of the `GenerateNAID` node and provide a source image.

![image](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/15ff8961-4f6b-4f23-86bf-34b86ace45c0)

**Note:** The width and height of the source image will be resized to the generation size.

### Inpainting

Connect an `InpaintingOptionNAID` node to the `GenerateNAID` node and provide a source image and a mask.

![image](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/5ed1ad77-b90e-46be-8c37-9a5ee0935a3d)

**Note:** Both the source image and mask will be automatically resized to fit the generation size.

### Vibe Transfer

Connect a `VibeTransferOptionNAID` node to the `GenerateNAID` node and provide a reference image to transfer its style and feel.

![Comfy_workflow](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/8c6c1c2e-f29d-42a1-b615-439155cb3164)

You can also chain it with other options, like Img2Img.

![image](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/acf0496c-8c7c-48f4-9530-18e6a23669d5)

#### Multiple Vibe Transfer

Connect multiple `VibeTransferOptionNAID` nodes to combine their influences.

![preview_vibe_2](https://github.com/user-attachments/assets/2d56c0f7-bcd5-48ff-b436-012ea43604fe)

### Character Reference

Use the `CharacterReferenceOptionNAID` node to guide the generation using a single reference image for character identity and/or style.

![image](https://github.com/user-attachments/assets/9f21eff1-163e-49f4-9e4a-85ea25b29988)

-   **style_aware:** If enabled, it attempts to copy both the character's features and the artistic style.
-   **fidelity:** Controls how strictly the generation should adhere to the reference image. The developers state that `primary_strength` is always kept at `1.0`, while `secondary_strength` is calculated as $1.0 - \text{fidelity}$.

**Note:** The reference image will be automatically letterboxed to an accepted NAI canvas size to preserve its aspect ratio.

### ModelOption

The default model of the `GenerateNAID` node is `nai-diffusion-4-5-full`. To change the model, connect a `ModelOptionNAID` node.

Available V4+ models include:
- `nai-diffusion-4-curated-preview`
- `nai-diffusion-4-full`
- `nai-diffusion-4-5-curated`
- `nai-diffusion-4-5-full`
- `nai-diffusion-5-curated`
- `nai-diffusion-5-full`

**Note:** NovelAI Diffusion V5 launched on 2026-08-21. The `nai-diffusion-5-*` identifiers above follow NovelAI's established naming convention but have not been independently confirmed against the live API yet — if generation fails with an unrecognized model error, please open an issue so the identifier can be corrected.

![ModelOption](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/0b484edb-bcb5-428a-b2af-1372a9d7a34f)

### NetworkOption

You can set timeout and retry options using the `NetworkOption` node. You can also set `ignore_errors` to prevent the queue from stopping on an API error; if an error occurs, a blank 1x1 image will be output.

![preview_network](https://github.com/user-attachments/assets/d82b0ff2-c57c-4870-9024-8d78261a8fea)

**Note:** If you set the timeout too short, you may not receive an image but could still be charged Anlas.

### Anlas Tracker

This extension now includes Anlas tracking to monitor your usage.

**1. Console Output (Automatic)**
All generation and director tool nodes will automatically print your Anlas balance before and after the operation in the console where you launched ComfyUI.
```
[NovelAI] Anlas (pre-gen): 10000
[NovelAI] Generation cost: 20 Anlas
[NovelAI] Anlas (post-gen): 9980
```

**2. Visual Node**
A new node, **`Anlas Tracker ✒️🅝🅐🅘`**, is available in the `NovelAI/utils` category. You can use it to display your current Anlas balance directly in your workflow.

Connect its `anlas_string` output to a display node (e.g., "Show Text" from the [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui)) to see the value. Use the `trigger` input to control when the balance is checked.

![image](https://github.com/user-attachments/assets/387799cd-36bd-4b5d-9746-bfd74db09f74) <!-- Placeholder for actual image -->

### PromptToNAID

ComfyUI uses `()` or `(word:weight)` for emphasis, while NovelAI uses `{}` and `[]`. This node, found in `NovelAI/utils`, converts ComfyUI's prompt syntax to NovelAI's.

![image](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/25c48350-7268-4d6f-81fe-9eb080fc6e5a)

### Director Tools

![image](https://github.com/user-attachments/assets/e205a51e-59dc-4d5a-94c8-29715ed98739)

You can find director tools like `LineArtNAID`, `EmotionNAID`, and `RemoveBGNAID` in the `NovelAI/director_tools` category.

![augment_example](https://github.com/user-attachments/assets/5833e9fb-f92e-4d53-9069-58ca8503a3e7)

### V4 / V4.5 / V5 Support

The nodes fully support NAI's V4, V4.5 and V5 model architecture.

#### V4 Prompt Handling

Two new nodes have been added in `NovelAI/v4` for V4/V4.5/V5 prompt handling:

- **`V4BasePrompt`**: Handles the positive prompt.
- **`V4NegativePrompt`**: Handles the negative prompt.

#### Example V4 / V4.5 / V5 Workflow

Here's a basic setup for a V4/V4.5/V5 model:

```
V4BasePrompt -----> positive
                           GenerateNAID
V4NegativePrompt -> negative
```

**Note:** Basic `img2img`, `vibe transfer` and `inpainting` functionality works with V4/V4.5/V5.

### Character Prompt (Multi-Character Text Prompting)

Connect one or more `CharacterPromptOptionNAID` nodes to the `option` input of `GenerateNAID` to give individual characters their own positive/negative captions, mirroring NAI's multi-character prompting feature (V4+). Chain multiple nodes together, one per character, the same way you would chain `VibeTransferOptionNAID` nodes.

- **positive / negative:** Per-character captions, merged into the request's `char_captions`.
- **use_position / x / y:** Enable to pin the character to a specific normalized position (`0.0`-`1.0`) instead of letting the model place it automatically. If any chained character enables this, positioning is used for the whole group.
- **use_order:** Treat the connection order of the chained character nodes as a priority hint for overlapping characters. If any chained character enables this, it applies to the whole group.

**Note:** This is independent of `CharacterReferenceOptionNAID`, which guides generation from a reference *image* rather than text.
