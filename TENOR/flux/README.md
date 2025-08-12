# Training FLUX.1-dev with CoMPaSS

> This code is derived from the awesome [x-flux] repository.

## Prerequisites

- Set up the SCOP dataset.
- Download [black-forest-labs/FLUX.1-dev]:
  ```bash
  huggingface-cli download black-forest-labs/FLUX.1-dev
  ```
- (Needed for inference) Download the reference weights ([instructions](../README.md#reference-weights))

## Training

Update `data_config.root` in [./train_configs/compass.yaml] to the directory containing
the prepared SCOP dataset, then run:

```bash
accelerate launch compass_train_lora.py --config train_configs/compass.yaml
```

The default config will save a LoRA checkpoint every 2,000 training steps, and sample
images every 2,500 steps.

## Inference

```python
from src.flux.xflux_pipeline import XFluxPipeline

# load weights
pipe = XFluxPipeline("flux-dev", device="cuda")

# load lora, with TENOR enabled
pipe.set_lora(
  use_tenor=True,
  local_path="path/to/lora.safetensors",
  lora_weight=1.0,  # use full weight
)

# sample an image
image = pipe("a photo of a bird below a skateboard")

# save the image for viewing
image.save("bird-below-skateboard.png")
```

[x-flux]: <https://github.com/XLabs-AI/x-flux>
[./train_configs/compass.yaml]: <./train_configs/compass.yaml>
[black-forest-labs/FLUX.1-dev]: <https://huggingface.co/black-forest-labs/FLUX.1-dev>

<!-- vim: set ts=2 sts=2 sw=2 et: -->
