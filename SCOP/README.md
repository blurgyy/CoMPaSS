# Spatial Constraints-Oriented Pairing (SCOP) Data Engine

This directory contains the official implementation of the SCOP data engine.

## Usage

### Obtaining Images

Go to the repository root (i.e. parent directory of this directory) and run:

```bash
# make sure CWD is the repository root
python3 -m SCOP --coco-root /path/to/coco2017 --output-dir ./scop-coco2017
```

The `--coco-root` directory should contain `annotations/` and `train2017/`.

On successful processing of COCO2017, the output directory (`scop-coco2017` in this example) should
have a `metadata.jsonl` file (28,028 lines) and an `images/` folder (15,426 images).

### Obtaining Object Masks (Optional for FLUX.1, required for Stable Diffusion 1.4/1.5/2.1)

Install [SAM2] per the [official instructions][SAM2] (if you ran the [../setup_env.sh](../setup_env.sh)
script to set up your environment, SAM2 should have already been installed), then run:

```bash
# make sure CWD is the repository root
python3 -m SCOP.process_masks ./scop-coco2017
```

[SAM2]: <https://github.com/facebookresearch/sam2>

<!-- vim: set ts=2 sts=2 sw=2 et: -->
