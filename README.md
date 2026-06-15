<div align="center">

# Unified Open-World Segmentation with Multi-Modal Prompts

[Yang Liu](https://scholar.google.com/citations?user=9JcQ2hwAAAAJ&hl=en)<sup>1*</sup>,
Yufei Yin<sup>2*</sup>,
Chenchen Jing<sup>3</sup>,
Muzhi Zhu<sup>1</sup>,
[Hao Chen](https://stan-haochen.github.io/)<sup>1</sup>,
Yuling Xi<sup>1</sup>,
Bo Feng<sup>4</sup>,
Hao Wang<sup>4</sup>,
Shiyu Li<sup>4</sup>,
[Chunhua Shen](https://cshen.github.io/)<sup>1</sup>

<sup>1</sup>[Zhejiang University](https://www.zju.edu.cn/english/),
<sup>2</sup>Hangzhou Dianzi University,
<sup>3</sup>Zhejiang University of Technology,
<sup>4</sup>Apple

<p>
  <a href="https://www.modelscope.cn/models/zzzmmz/COSINE-Public-Weights">
    <img alt="ModelScope Weights" src="https://img.shields.io/badge/ModelScope-Weights-1677ff">
  </a>
</p>

<img width="800" alt="COSINE teaser" src="figs/teaser.png">

</div>

## Overview

COSINE is a unified open-world segmentation model for open-vocabulary segmentation and in-context segmentation with multi-modal prompts. It uses foundation-model features from the input image and text/visual prompts, then aligns them through a segmentation decoder to predict prompt-specific masks.

This repository is organized for public release and reproduction. Local datasets, checkpoints, and generated outputs should stay outside git under the paths listed below.

## Setup

```bash
conda create --name cosine python=3.9.17
conda activate cosine

pip install torch==2.0.1 torchvision==0.15.2
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

pip install -r requirements.txt
```

Optional DINOv2 speedup:

```bash
pip install xformers==0.0.21 torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu117
```

## Directory Layout

```text
datasets/                 # common datasets for FSS, RefSeg, VOS, training
models/                   # pretrained backbones and COSINE checkpoints
outputs/                  # logs, predictions, visualizations
cosine/                   # shared COSINE implementation package
inference_fsod/datasets/  # FSOD datasets used by detectron2-style configs
inference_fsod/models/    # optional FSOD-local checkpoint links/copies
inference_fsod/outputs/   # FSOD outputs
```

The scripts default to these relative paths. You can override checkpoint roots in shell scripts with:

```bash
WEIGHT_ROOT=/path/to/cosine-weights bash scripts/fss/eval_fss_coco20i.sh
```

## Weights

Download the DINOv2 ViT-L pretrained weight and place it at:

```text
models/dinov2_vitl14_pretrain.pth
```

COSINE checkpoints are hosted on [ModelScope](https://www.modelscope.cn/models/zzzmmz/COSINE-Public-Weights) and are expected under `models/cosine/` using the public checkpoint directory names. See [MODEL_ZOO.md](MODEL_ZOO.md) for the checkpoint map.

With ModelScope access, download the release checkpoint files and place the `weights/` contents under `models/cosine/`:

```bash
MODELSCOPE_TOKEN=... bash scripts/download_weights_modelscope.sh
bash scripts/check_required_assets.sh --weights-only
```

The token is optional when your ModelScope CLI is already authenticated.

## Evaluation

Task-specific data layouts are documented in each subdirectory:

- Few-shot semantic segmentation: [inference_fss/EVALUATION.md](inference_fss/EVALUATION.md)
- Few-shot instance segmentation: [inference_fsod/EVALUATION.md](inference_fsod/EVALUATION.md)
- Video object segmentation: [inference_vos/EVALUATION.md](inference_vos/EVALUATION.md)

The consolidated dataset layout is documented in [DATASETS.md](DATASETS.md).
Before running evaluation, check local assets with:

```bash
bash scripts/check_required_assets.sh
```

Representative entry points:

```bash
bash scripts/fss/eval_fss_coco20i.sh
bash scripts/refseg/eval_referseg_dist_ms.sh
bash scripts/vos/eval_vos_d17_ms.sh

cd inference_fsod
bash scripts/coco_ms.sh
bash scripts/lvis_ms_fcclip.sh
```

See [REPRODUCTION.md](REPRODUCTION.md) for the current reproduction checklist.
The script inventory is tracked in [EVALUATION_SCRIPTS.md](EVALUATION_SCRIPTS.md).
The source layout and the role of `cosine/` are documented in [SOURCE_LAYOUT.md](SOURCE_LAYOUT.md).
For a quick functional check before full metrics, use the bounded smoke options
recorded in [REPRODUCTION.md](REPRODUCTION.md#verified-minimal-smoke).

## Training

Training commands and dataset preparation notes are in [TRAINING.md](TRAINING.md). The default training scripts use `datasets/`, `models/`, and `outputs/` unless explicit command-line paths are provided.

## License

For academic use, this project is licensed under the [2-clause BSD License](LICENSE). For commercial use, please contact [Chunhua Shen](mailto:chhshen@gmail.com).
