<div align="center">

# HiAnimal

### Towards High-fidelity and Animatable Mesh Reconstruction<br>from Single-view In-the-Wild Animal Images

**Huimin Zhang · Zhongjin Luo · Kenkun Liu · Xihe Yang · Haolin Liu · Weikai Chen · Xiaoguang Han**

[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b?style=flat-square)](./TVCG_HiAnimal.pdf)
[![Video](https://img.shields.io/badge/Video-MP4-2f6f68?style=flat-square)](./tvcg_video_hianimal.mp4)
[![Code](https://img.shields.io/badge/Code-Coming_Soon-lightgrey?style=flat-square)](#installation)

<a href="./tvcg_video_hianimal.mp4">
  <img src="./assets/teaser.gif" width="100%" alt="HiAnimal teaser: single-view animal images and reconstructed animatable 3D meshes">
</a>

<sub>Click the teaser to view the full-quality video.</sub>

</div>

## About

**HiAnimal** reconstructs high-fidelity and animatable 3D quadruped meshes from a single in-the-wild image. It combines a pixel-aligned implicit prior with the parametric SMAL model to preserve fine geometric details while producing clean meshes with consistent topology for animation and simulation.

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
