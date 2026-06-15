# Evaluation Scripts

This file identifies the evaluation entry points to use for public reproduction. Some older top-level scripts are kept for history, but the grouped scripts below are the preferred starting point.

## Preferred Scripts

| Task | Dataset / setting | Script | Evaluator | Checkpoint |
| --- | --- | --- | --- | --- |
| FSS, multi-scale | COCO-20i | `scripts/fss/eval_fss_coco20i_ms.sh` | `tools/eval_fss_ms.py` | `mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin` |
| FSS, multi-scale | LVIS-92i | `scripts/fss/eval_fss_lvis_ms1.sh`, `scripts/fss/eval_fss_lvis_ms2.sh` | `tools/eval_fss_ms.py` | `mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin` |
| FSS, single-scale | COCO-20i | `scripts/fss/eval_fss_coco20i.sh` | `tools/eval_fss.py` | `cosine_fd1_md6_lr1e-4_ep50/pytorch_model.bin` |
| FSS, single-scale | LVIS-92i | `scripts/fss/eval_fss_lvis1.sh`, `scripts/fss/eval_fss_lvis2.sh`, `scripts/fss/eval_fss_lvis3.sh` | `tools/eval_fss.py` | `cosine_fd1_md6_lr1e-4_ep50/pytorch_model.bin` |
| Referring segmentation | RefCOCO / RefCOCO+ / RefCOCOg | `scripts/refseg/eval_referseg_dist_ms.sh` | `tools/eval_referseg_ms.py` | `mscosine_refseg_fd1_md6_lr1e-4_ep50/pytorch_model_19ep/pytorch_model.bin` |
| VOS | DAVIS 2017 | `scripts/vos/eval_vos_d17_ms.sh` | `tools/eval_vos_ms.py` plus DAVIS official evaluator | `mscosine_vos_fd1_md6_lr1e-4_ep50/pytorch_model/pytorch_model_24ep/pytorch_model.bin` |
| VOS | YouTube-VOS 2019 | `scripts/vos/eval_vos_y19_ms.sh` | `tools/eval_vos_ms.py` | `mscosine_vos_fd1_md6_lr1e-4_ep50/pytorch_model/pytorch_model_24ep/pytorch_model.bin` |
| FSOD | COCO | `inference_fsod/scripts/coco_ms.sh` | `inference_fsod/tools/preprocess.py`, `inference_fsod/tools/test.py` | `mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin` |
| FSOD | LVIS `MSCOSINE_all_fc` / `vis_text` | `inference_fsod/scripts/lvis_ms_fcclip.sh` | `inference_fsod/tools/test.py` | `mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin` |

## Runtime Environment

The preferred shell scripts source `scripts/common.sh` and can be redirected
without editing the files:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PYTHON_BIN` | `python` | Python executable or environment launcher |
| `GPU` | `0` | Value assigned to `CUDA_VISIBLE_DEVICES` |
| `WEIGHT_ROOT` | `<repo>/models/cosine` | Root that contains released COSINE checkpoints |
| `DATA_ROOT` | `<repo>/datasets` | Root for RefSeg and VOS datasets |
| `FSS_DATA_ROOT` | `$DATA_ROOT/fss` | Root for FSS datasets |
| `OUTPUT_ROOT` | `<repo>/outputs` | Root for FSS, RefSeg, and VOS outputs |
| `FSOD_OUTPUT_ROOT` | `<repo>/inference_fsod/outputs` | Root for FSOD outputs |
| `NUM_GPUS` | task-specific | Distributed GPU count for RefSeg/FSOD scripts |

Example:

```bash
GPU=0 WEIGHT_ROOT=/path/to/cosine-weights DATA_ROOT=/path/to/datasets \
  bash scripts/fss/eval_fss_coco20i_ms.sh
```

## Bounded Smoke Options

The main evaluators expose default-off options for a quick functional check
before launching full reproduction jobs:

| Task | Evaluator | Smoke option |
| --- | --- | --- |
| FSS, multi-scale | `tools/eval_fss_ms.py` | `--max-samples 1` |
| Referring segmentation | `tools/eval_referseg_ms.py` | `--max-samples 1` |
| VOS | `tools/eval_vos_ms.py` | `--max-videos 1 --max-frames 2` |
| FSOD | `inference_fsod/tools/test.py` | `--smoke-model-only` |

These flags are intended for import, path, checkpoint, dataset, and model-load
validation. They are not substitutes for the full metric scripts above.

## Notes

- VOS MS scripts currently sweep epochs `9 14 19 24`; use `24ep` as the canonical paper-reproduction checkpoint unless an ablation explicitly needs the sweep.
- `scripts/refseg/*_vis*.sh` are visualization/demo variants, not the main metric entry point.
- `inference_fsod/scripts/lvis_ms_fcclip.sh` runs the `vis_text` block. It creates the FSOD `checkpoint.pth` query cache with `tools/preprocess.py` when the cache is missing; set `FORCE_PREPROCESS=1` to rebuild it.
- The FSOD LVIS evaluator includes a local compatibility shim for recent NumPy versions where `np.float` was removed from the upstream `lvis` package.
- Older top-level scripts are kept for continuity, but the grouped scripts above are the preferred public reproduction entry points.
