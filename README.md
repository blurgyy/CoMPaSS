# CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models

Gaoyang Zhang<sup>1,2</sup>, Bingtao Fu<sup>2</sup>, [Qingnan Fan](https://fqnchina.github.io)<sup>2</sup>, [Qi Zhang](https://qzhang-cv.github.io)<sup>2</sup>, Runxing Liu<sup>2</sup>, Hong Gu<sup>2</sup>, Huaqi Zhang<sup>2</sup>, Xinguo Liu<sup>1,✉</sup>

<sup>1</sup> State Key Laboratory of CAD&CG, Zhejiang University  
<sup>2</sup> vivo Mobile Communication Co. Ltd  
<sup>✉</sup> Corresponding author

[[Project page](https://compass.blurgy.xyz)]
[[arXiv](https://arxiv.org/abs/2412.13195)]

## Abstract

Text-to-image diffusion models excel at generating photorealistic images, but commonly struggle to
render accurate spatial relationships described in text prompts.  We identify two core issues
underlying this common failure:  1) the ambiguous nature of spatial-related data in existing
datasets, and 2) the inability of current text encoders to accurately interpret the spatial
semantics of input descriptions.
We address these issues with CoMPaSS, a versatile training framework that enhances spatial
understanding of any T2I diffusion model.
CoMPaSS solves the ambiguity of spatial-related data with the **S**patial
**C**onstraints-**O**riented **P**airing (SCOP) data engine, which curates spatially-accurate
training data through a set of principled spatial constraints.
To better exploit the curated high-quality spatial priors, CoMPaSS further introduces a **T**oken
**EN**coding **OR**dering (TENOR) module to allow better exploitation of high-quality spatial priors,
effectively compensating for the shortcoming of text encoders.
Extensive experiments on four popular open-weight T2I diffusion models covering both UNet- and
MMDiT-based architectures demonstrate the effectiveness of CoMPaSS by setting new state-of-the-arts
with substantial relative gains across well-known benchmarks on spatial relationships generation,
including
[VISOR](https://github.com/microsoft/VISOR) (_+98%_),
[T2I-CompBench Spatial](https://github.com/Karine-Huang/T2I-CompBench) (_+67%_), and
[GenEval Position](https://github.com/djghosh13/geneval) (_+131%_).

## Citation

```bibtex
@article{zhang2024compass,
  title={CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models},
  author={Zhang, Gaoyang and Fu, Bingtao and Fan, Qingnan and Zhang, Qi and Liu, Runxing and Gu, Hong and Zhang, Huaqi and Liu, Xinguo},
  journal={arXiv preprint arXiv:2412.13195},
  year={2024}
}
```
