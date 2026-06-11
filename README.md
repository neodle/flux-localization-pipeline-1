# FLUX Localization Pipelinecccc

Automatic Object Transformation in Image via FLUX Inpainting with Multi-LoRA Condition Decomposition

<img width="1280" height="720" alt="pipeline" src="https://github.com/user-attachments/assets/c423934a-5fcf-4ca2-8542-b021fe7e9500" />
<img width="1051" height="429" alt="flux_image" src="https://github.com/user-attachments/assets/072caff7-162b-47ad-aaf1-7ec2e4877be7" />

---

## Overview

This repository provides the official implementation of:

> **FLUX 인페인팅과 다중 LoRA 조건 분리를 이용한 영상 객체 자동 변환**

The proposed framework automatically transforms visual objects in images and video frames while preserving:

- background consistency
- object identity
- lighting coherence
- spatial alignment

Our method combines:

- Grounding DINO for object localization
- Segment Anything Model (SAM) for precise mask extraction
- FLUX Inpainting
- Multi-LoRA conditional decomposition

The pipeline separates:

- Fill condition (background)
- Subject condition (reference object)
- Denoising condition (generation stabilization)

to achieve robust object-level localization and replacement.

---

## Pipeline

The framework consists of four stages:

1. Object Localization (Grounding DINO)
2. Mask Extraction (SAM)
3. Condition Construction
4. FLUX Multi-LoRA Inpainting

### Multi-LoRA Composition

```math
\Delta \theta_{LoRA}
=
\Delta \theta_{fill}
+
\Delta \theta_{subj}
+
\Delta \theta_{denoise}
```

- Fill LoRA:
  preserves scene-level texture and lighting

- Subject LoRA:
  preserves object identity and fine details

- Denoising LoRA:
  stabilizes iterative diffusion generation

---

## Features

- Automatic object localization
- Precise object segmentation
- Object-aware inpainting
- Multi-condition controllable generation
- Background-preserving object replacement
- Visual localization for media contents
- Robust under occlusion conditions

---

# Environment Setup

Create conda environment:

```bash
conda create -n flux-localization python=3.10

conda activate flux-localization
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Required External Libraries

This project is built on top of the following frameworks:

- FLUX.1-schnell
- UniCombine
- Grounding DINO
- Segment Anything Model (SAM)

Please install and configure these libraries locally before running the pipeline.

---

## 1. Install Grounding DINO

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git

cd GroundingDINO

pip install -e .

cd ..
```

---

## 2. Install Segment Anything Model (SAM)

```bash
git clone https://github.com/facebookresearch/segment-anything.git

cd segment-anything

pip install -e .

cd ..
```

---

## 3. Install UniCombine

```bash
git clone https://github.com/frank-xwang/UniCombine.git
```

---

## 4. Download FLUX.1-schnell

Download FLUX model from:

https://huggingface.co/black-forest-labs/FLUX.1-schnell

---

# Local Environment Configuration

The original experimental environment was organized as follows:

```bash
workspace/
│
├── UniCombine/
│   ├── assets/
│   ├── ckpt/
│   ├── demo_Condition_LoRA/
│   ├── examples/
│   ├── output/
│   └── src/
│
├── bb_generate/
│   ├── demo/
│   ├── groundingdino/
│   ├── datasets/
│   ├── models/
│   ├── util/
│   └── weights/
│
└── mask_generate/
    ├── assets/
    ├── demo/
    ├── notebooks/
    ├── scripts/
    └── segment_anything/
```

The files provided in this repository correspond only to custom implementation and modified experimental modules used in our paper.

# Our Custom Modules

This repository only contains our custom implementation and modified experimental modules.

External frameworks such as:

- Grounding DINO
- Segment Anything Model (SAM)
- FLUX.1-schnell
- UniCombine

must be installed separately in the local environment.

The provided scripts are designed to be integrated into the original framework structure.

---

## Bounding Box Generation

Directory:

```bash
bb_generate/
```

Functions:

- Grounding DINO inference
- object localization
- bounding box extraction

Files:

```bash
bb_generate/
├── grounded_sam_beverage.py
├── grounded_sam_occlusion.py
├── grounded_sam_origin.py
└── requirements.txt
```

---

## Mask Generation

Directory:

```bash
mask_generate/
```

Functions:

- SAM-based segmentation
- binary mask generation
- mask refinement

Files:

```bash
mask_generate/
├── target_sam.py
└── target_sam_binary.py
```

---

## Multi-LoRA FLUX Integration

Directory:

```bash
UniCombine/examples/custom/
```

Functions:

- Fill LoRA conditioning
- Subject LoRA conditioning
- Denoising LoRA stabilization
- conditional FLUX generation

Files:

```bash
UniCombine/examples/custom/
├── Image_resize.py
├── background_occ.py
├── force_composite.py
├── inference.py
├── inference.txt.txt
├── prepare_inputs.py
├── requirements.txt
└── update_inputs.py
```

---

## Requirements

```bash
torch>=2.0.0
torchvision
diffusers
transformers
accelerate
opencv-python
numpy
pillow
matplotlib
segment-anything
groundingdino-py
einops
safetensors
tqdm
```

---

# Project Structure

```bash
flux-localization-pipeline/
│
├── UniCombine/
│   └── examples/custom/
│       ├── Image_resize.py
│       ├── background_occ.py
│       ├── force_composite.py
│       ├── inference.py
│       ├── inference.txt.txt
│       ├── prepare_inputs.py
│       ├── requirements.txt
│       └── update_inputs.py
│
├── bb_generate/
│   ├── grounded_sam_beverage.py
│   ├── grounded_sam_occlusion.py
│   ├── grounded_sam_origin.py
│   └── requirements.txt
│
└── mask_generate/
    ├── target_sam.py
    └── target_sam_binary.py
```

---

# Model Checkpoints

Download required checkpoints manually.

| Model | Source |
|---|---|
| Grounding DINO | https://github.com/IDEA-Research/GroundingDINO |
| SAM ViT-H | https://github.com/facebookresearch/segment-anything |
| FLUX.1-schnell | https://huggingface.co/black-forest-labs/FLUX.1-schnell |

---

# Inference Pipeline

The overall inference process consists of four stages:

1. Target object localization
2. SAM-based mask extraction
3. Binary mask generation
4. FLUX Multi-LoRA inpainting

---

## Step 1. Object Localization + SAM Mask Extraction

Inside the `bb_generate/` directory, the target object is first detected using:

```bash
grounded_sam_bvg.py
```

This step performs:

- Grounding DINO object detection
- SAM segmentation
- bounding box extraction

Outputs:

- detected bounding box coordinates
- SAM mask image

Example:

```bash
python bb_generate/grounded_sam_bvg.py
```

---

## Step 2. Binary Mask Generation

Inside the `mask_generate/` directory, the generated bounding box coordinates and SAM mask image are used as inputs for:

```bash
cola_sam_box.py
```

This step generates a binary mask image:

- Target object = white (255)
- Background = black (0)

Output:

- binary target mask image

Example:

```bash
python mask_generate/cola_sam_box.py
```

---

## Step 3. Input Preprocessing

Inside the `UniCombine/examples/custom/` directory:

### 1. Resize Input Images

Run:

```bash
Image_resize.py
```

This script resizes:

- target image
- target binary mask

to:

```text
512 × 512 resolution
```

(maximum size: 512)

Example:

```bash
python UniCombine/examples/custom/Image_resize.py
```

---

### 2. Generate Background Image

Run:

```bash
prepare_inputs.py
```

Inputs:

- target image
- target binary mask image

This step generates:

- background image for FLUX Fill conditioning

Example:

```bash
python UniCombine/examples/custom/prepare_inputs.py
```

---

## Step 4. FLUX Multi-LoRA Inference

Finally, run the command provided in:

```bash
inference.txt
```

to execute:

```bash
inference.py
```

This step performs:

- FLUX inpainting
- Fill LoRA conditioning
- Subject LoRA conditioning
- Denoising LoRA stabilization

Final output:

- transformed object image
- background-preserved localization result

Example:

```bash
python UniCombine/examples/custom/inference.py
```

---

# Full Inference Flow

```text
Target Image
    ↓
grounded_sam_bvg.py
    ↓
Bounding Box + SAM Mask
    ↓
cola_sam_box.py
    ↓
Binary Target Mask
    ↓
Image_resize.py
    ↓
prepare_inputs.py
    ↓
Background Image Generation
    ↓
inference.py
    ↓
Final FLUX Localization Output
```


## Experimental Settings

- Resolution: 512 × 512
- Diffusion steps: 6
- GPU: NVIDIA RTX 4090
- OS: Ubuntu 22.04

Recommended LoRA weights:

| LoRA Type | Recommended Weight |
|---|---|
| Fill | 0.6 ~ 0.9 |
| Subject | 1.0 ~ 1.4 |
| Denoising | 1.0 |

---

## Results

### Object Transformation

- Preserves object position
- Maintains lighting consistency
- Supports realistic replacement

### Occlusion Robustness

The proposed method remains stable even under partial object occlusion.

### Media Localization

Applicable to:

- advertisement localization
- virtual PPL
- media remastering
- global content adaptation

---

## Citation

```bibtex
@inproceedings{flux_localization_2026,
  title={Automatic Object Transformation in Image via FLUX Inpainting with Multi-LoRA Condition Decomposition},
  author={Kang, Minsoo and Jeong, Ayoung and Kim, Namho and Kim, Junhwa},
  year={2026}
}
```

---

## Acknowledgement

This work was supported by the IITP MSIT SW-centered University Program (2024-0-00047).

---

## Paper

**Korean Title**  
> FLUX 인페인팅과 다중 LoRA 조건 분리를 이용한 영상 객체 자동 변환

**English Title**  
> Automatic Object Transformation in Image via FLUX Inpainting with Multi-LoRA Condition Decomposition

**Conference**  
> 2026 Summer Conference of the Korean Institute of Broadcast and Media Engineers  
> 방송미디어공학회 2026 하계 학술대회

Implementation repository:

[flux-localization-pipeline](https://github.com/flux-pipeline-JK/flux-localization-pipeline?utm_source=chatgpt.com)

---

# Notes

This repository does NOT include:

- external repositories
- pretrained checkpoints
- FLUX base models

Only custom implementation and modified modules are provided.
