# Training UNet-based Stable Diffusion Models with CoMPaSS

> This code is derived from the awesome [TokenCompose] repository.

## Prerequisites

- Set up the SCOP dataset.
- Download the base models:
  - For SD1.4, download [CompVis/stable-diffusion-v1-4]
  - For SD1.5, download [runwayml/stable-diffusion-v1-5]
  - For SD2.1, download [stabilityai/stable-diffusion-2-1]
  E.g.,
  ```bash
  huggingface-cli download CompVis/stable-diffusion-v1-4
  ```
- (Needed for inference) Download the reference weights ([instructions](../README.md#reference-weights))

## Training

Set the `TRAIN_DATA_DIR` variable in the training script (located in the [./scripts]
directory) to the directory containing the prepared SCOP dataset, then run the script.
For example, to train SD1.4:

```bash
bash ./scripts/sd14.sh
```

## Inference

Inference follows standard practice for the `diffusers` library.  Replace the UNet in
the base model with our trained UNet (showing SD1.5 as an example):

```python
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch

# load our UNet
compass_unet = UNet2DConditionModel.from_pretrained(
  "path/to/checkpoint-24000",
  subfolder="unet",
  torch_dtype=torch.float16,
)

# NB: register TENOR's attention processor
from src.attn_utils import register_pe_processor
compass_unet, _ = register_pe_processor(compass_unet, "absolute")

# load the pipeline with the loaded unet
pipe = StableDiffusionPipeline.from_pretrained(
  "runwayml/stable-diffusion-v1-5",
  unet=compass_unet,
  torch_dtype=torch.float16,
).cuda()

# sample an image
image = pipe("a photo of a motorcycle to the right of a bear").images[0]

# save the image for viewing
image.save("motorcycle-right-bear.png")
```

[TokenCompose]: <https://github.com/mlpc-ucsd/TokenCompose>
[./scripts]: <./scripts>

[CompVis/stable-diffusion-v1-4]: <https://huggingface.co/CompVis/stable-diffusion-v1-4>
[runwayml/stable-diffusion-v1-5]: <https://huggingface.co/runwayml/stable-diffusion-v1-5>
[stabilityai/stable-diffusion-2-1]: <https://huggingface.co/stabilityai/stable-diffusion-2-1>

<!-- vim: set ts=2 sts=2 sw=2 et: -->
