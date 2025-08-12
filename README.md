# CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models

**\[[Project Page]\]
\[[arXiv]\]**

> [Gaoyang Zhang], Bingtao Fu, [Qingnan Fan], [Qi Zhang], Runxing Liu, Hong Gu, Huaqi Zhang, Xinguo Liu  
> ICCV 2025

## TL; DR

CoMPaSS enhances the spatial understanding of existing text-to-image diffusion models, enabling
them to generate images that faithfully reflect spatial configurations specified in the
text prompt.

![teaser](./assets/teaser.avif)

## Setting up Environment

We manage our python environment with [uv], and provide a convenient script for setting
up the environment at [setup_env.sh](./setup_env.sh).
Running this script will create a subdirectory `.venv/` in the project root.  To enable
it, run `source .venv/bin/activate` after the environment is set up:

```bash
# install requirements into .venv/
bash ./setup_env.sh

# activate the environment
source .venv/bin/activate
```

## Trying out CoMPaSS

> [!NOTE]
> For training, SCOP and TENOR are both required.  
> For generating images from text, only TENOR and the reference weights are needed.

### Reference Weights

We provide the reference weights used to report all metrics in our paper on Hugging
Face 🤗.
We recommend trying out the FLUX.1-dev weights as it is a Rank-16 LoRA which is only
50MB in size.

| Model | Link |
|:-----:|:-----:|
| FLUX.1-dev | <https://huggingface.co/blurgy/CoMPaSS-FLUX.1> |
| SD1.4 | <https://huggingface.co/blurgy/CoMPaSS-SD1.4> |
| SD1.5 | <https://huggingface.co/blurgy/CoMPaSS-SD1.5> |
| SD2.1 | <https://huggingface.co/blurgy/CoMPaSS-SD2.1> |

### The SCOP dataset

We provide full instructions for replicating the SCOP dataset (28,028 object pairs among
15,426 images) in the [SCOP](./SCOP) directory.  Check out its [README](./SCOP/README.md)
to get started.

### The TENOR Module

We provide both training and inference instructions for using our TENOR module in the
[TENOR](./TENOR) directory.
MMDiT-based models (e.g., FLUX.1-dev) and UNet-based models (e.g., SD1.5) are both
supported.  Check out their respective instructions to get started:
- [Instructions for FLUX.1-dev](./TENOR/flux/README.md)
- [Instructions for SD1.4, SD1.5, and SD2.1](./TENOR/sd/README.md)

## Citation

```bibtex
@inproceedings{zhang2025compass,
  title={CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models},
  author={Zhang, Gaoyang and Fu, Bingtao and Fan, Qingnan and Zhang, Qi and Liu, Runxing and Gu, Hong and Zhang, Huaqi and Liu, Xinguo},
  booktitle={ICCV},
  year={2025}
}
```

[Gaoyang Zhang]: <https://github.com/blurgyy>
[Qingnan Fan]: <https://fqnchina.github.io>
[Qi Zhang]: <https://qzhang-cv.github.io>

[Project Page]: <https://compass.blurgy.xyz>
[arXiv]: <https://arxiv.org/abs/2412.13195>

[uv]: <https://github.com/astral-sh/uv>

[TokenCompose]: <https://github.com/mlpc-ucsd/TokenCompose>
[x-flux]: <https://github.com/XLabs-AI/x-flux>

<!-- vim: set ts=2 sts=2 sw=2 et: -->
