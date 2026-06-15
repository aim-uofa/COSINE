# Source Layout

This repository keeps the original research code layout where possible, while documenting which directories are public entry points and which are shared implementation modules.

## Core Directories

| Path | Role |
| --- | --- |
| `tools/` | Training and task-level evaluation CLIs used by the shell scripts. |
| `scripts/` | Main training/evaluation shell entry points. The grouped subdirectories are preferred for public reproduction. |
| `cosine/` | Shared COSINE implementation package: dataset builders, training utilities, matcher/criterion code, pixel decoders, transformer decoders, and task model variants. |
| `inference_fss/` | Few-shot semantic segmentation data/model wrappers. |
| `inference_referseg/` | Referring segmentation wrappers. |
| `inference_vos/` | Video object segmentation wrappers. |
| `inference_fsod/` | Few-shot object/instance segmentation code and Detectron2-style configs. |
| `dinov2/` | Vendored DINOv2 code used by encoders and training/evaluation utilities. |
| `segment_anything/` | Vendored SAM code used by visual-prompt paths. |

## Why `cosine/` Stays

`cosine/` is required by the current training and inference code. It is not only a legacy baseline directory:

- Training entry points import `cosine.data.*`, `cosine.model.*`, and `cosine.utils.*`.
- Multi-scale FSS, RefSeg, and VOS modules reuse `cosine.model.pixel_decoder.msdeformattn`.
- FSOD FC-CLIP/COSINE paths import shared data/category helpers and model components.
- Visualization scripts also build models and datasets from `cosine/`.

Renaming or deleting `cosine/` would require a broad import refactor across training, FSS, RefSeg, VOS, FSOD, and visualization code. For the first public release, keep `cosine/` as a shared package and document its role explicitly.

## Local-Only Directories

These paths are intentionally ignored by git:

```text
datasets/
models/
outputs/
inference_fsod/datasets/
inference_fsod/models/
inference_fsod/outputs/
```
