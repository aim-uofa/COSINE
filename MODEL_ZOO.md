# Model Zoo

This file records the public checkpoint names expected by the release scripts.
Place released weights under `models/cosine/`, or set
`WEIGHT_ROOT=/path/to/cosine-weights` when running shell scripts.

The current public packaging target is ModelScope repo
`zzzmmz/COSINE-Public-Weights`.

## Public Checkpoints

Public checkpoint names keep only neutral model and training information:
architecture, feature decoder depth, mask decoder depth, learning rate, and
training epochs. Dataset-specific or ablation-specific internal labels are not
part of the release names.

| Public checkpoint directory | Typical use |
| --- | --- |
| `cosine_fd1_md6_lr1e-4_ep50` | Single-scale COSINE checkpoint for FSS, RefSeg single-scale, and VOS single-scale baselines |
| `mscosine_unified_fd1_md6_lr1e-4_ep50` | Multi-scale COSINE checkpoint for FSS and FSOD |
| `mscosine_refseg_fd1_md6_lr1e-4_ep50` | Multi-scale referring segmentation checkpoint |
| `mscosine_vos_fd1_md6_lr1e-4_ep50` | Multi-scale VOS checkpoint; canonical evaluation uses `pytorch_model_24ep` |

## Download Layout

The ModelScope repo stores release checkpoints as `weights/<public-checkpoint>/...`.
The download script fetches only the public release manifest and checkpoint
files, then copies `weights/` into the scripts' expected layout:

```bash
MODELSCOPE_TOKEN=... bash scripts/download_weights_modelscope.sh
bash scripts/check_required_assets.sh --weights-only
```

Set `MODELSCOPE_TOKEN` only when the local ModelScope CLI is not already logged
in.

## Task Checkpoint Map

| Task | Dataset / setting | Checkpoint path under `models/cosine/` |
| --- | --- | --- |
| FSS, multi-scale | COCO-20i / LVIS-92i | `mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin` |
| FSS, single-scale | COCO-20i / LVIS-92i | `cosine_fd1_md6_lr1e-4_ep50/pytorch_model.bin` |
| Referring segmentation | RefCOCO / RefCOCO+ / RefCOCOg | `mscosine_refseg_fd1_md6_lr1e-4_ep50/pytorch_model_19ep/pytorch_model.bin` |
| VOS | DAVIS 2017 / YouTube-VOS 2019 | `mscosine_vos_fd1_md6_lr1e-4_ep50/pytorch_model/pytorch_model_24ep/pytorch_model.bin` |
| FSOD | LVIS `MSCOSINE_all_fc`, `vis_text` | `mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin` |

Expected examples:

```text
models/cosine/
├── cosine_fd1_md6_lr1e-4_ep50/
│   └── pytorch_model.bin
├── mscosine_unified_fd1_md6_lr1e-4_ep50/
│   └── pytorch_model_49ep/pytorch_model.bin
├── mscosine_refseg_fd1_md6_lr1e-4_ep50/
│   └── pytorch_model_19ep/pytorch_model.bin
└── mscosine_vos_fd1_md6_lr1e-4_ep50/
    └── pytorch_model/pytorch_model_24ep/pytorch_model.bin
```
