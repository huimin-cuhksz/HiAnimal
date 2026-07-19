<div align="center">

# HiAnimal: Towards High-fidelity and Animatable Mesh Reconstruction from Single-view In-the-Wild Animal Images

**Huimin Zhang, Zhongjin Luo, Kenkun Liu, Xihe Yang, Haolin Liu, Weikai Chen, Xiaoguang Han**

[![Code](https://img.shields.io/badge/Code-Coming_Soon-lightgrey?style=flat-square)](#installation)

<img src="./assets/teaser.gif" width="100%" alt="HiAnimal teaser: single-view animal images and reconstructed animatable 3D meshes">

</div>

## About

**HiAnimal** reconstructs high-fidelity and animatable 3D quadruped meshes from a single in-the-wild image. It predicts a pixel-aligned Occupancy-UV field from an estimated normal map and aligns a parametric SMAL template with the reconstructed surface, producing image-aligned geometry with fine details, consistent topology, and transferable rigging for animation.

## Installation

> Code, pretrained models, and setup instructions will be released soon.

```bash
git clone https://github.com/huimin-cuhksz/HiAnimal.git
cd HiAnimal

# Environment setup — coming soon
```

## Inference

```bash
# Single-image reconstruction interface — coming soon
python inference.py \
  --image <input-image> \
  --output <output-directory>
```
