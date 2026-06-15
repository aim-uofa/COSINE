# Reproduction Checklist

This document tracks the public reproduction path. The goal is that a fresh clone can run evaluation without private `/home/...` paths or NAS symlinks.

See [EVALUATION_SCRIPTS.md](EVALUATION_SCRIPTS.md) for the full mapping from tasks to shell scripts.
See [SOURCE_LAYOUT.md](SOURCE_LAYOUT.md) for the source tree contract, including why `cosine/` is kept as a shared package.

## Required Local Artifacts

| Artifact | Default location |
| --- | --- |
| DINOv2 ViT-L pretrained weight | `models/dinov2_vitl14_pretrain.pth` |
| OpenCLIP ConvNeXt-L weight | `models/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/open_clip_pytorch_model.bin` |
| COSINE checkpoints | `models/cosine/...` |
| FSS datasets | `datasets/fss/...` |
| Referring segmentation datasets | `datasets/...` |
| VOS datasets | `datasets/vos/...` |
| FSOD datasets | `inference_fsod/datasets/...` |

Check the local layout before running smoke commands:

```bash
bash scripts/check_required_assets.sh
```

Download COSINE checkpoints from ModelScope when needed:

```bash
MODELSCOPE_TOKEN=... bash scripts/download_weights_modelscope.sh
```

The public checkpoint package is hosted at
`zzzmmz/COSINE-Public-Weights`. The download script fetches only the release
manifest and the four public checkpoint files, verifies `SHA256SUMS.txt` when
`sha256sum` is available, and copies the `weights/` tree into
`models/cosine/`.

## Smoke Commands

The commands below are full representative entry points. For a faster
functional check, the current evaluators also support bounded smoke options:

- `tools/eval_fss_ms.py --max-samples N`
- `tools/eval_referseg_ms.py --max-samples N`
- `tools/eval_vos_ms.py --max-videos N --max-frames M`
- `inference_fsod/tools/test.py --smoke-model-only`

These options are default-off and do not change full evaluation behavior.

Few-shot semantic segmentation:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval_fss_ms.py \
  --benchmark coco \
  --fold 0 \
  --nshot 1 \
  --weights models/cosine/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin \
  --log-root outputs/repro/fss/coco/1shot/fold0
```

Single-scale FSS uses the non-MS evaluator and checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval_fss.py \
  --benchmark coco \
  --fold 0 \
  --nshot 1 \
  --weights models/cosine/cosine_fd1_md6_lr1e-4_ep50/pytorch_model.bin \
  --log-root outputs/repro/fss_single/coco/1shot/fold0
```

Referring segmentation:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval_referseg_ms.py \
  --val_dataset "refcoco|unc|val" \
  --weights models/cosine/mscosine_refseg_fd1_md6_lr1e-4_ep50/pytorch_model_19ep/pytorch_model.bin \
  --output_dir outputs/repro/referseg/refcoco
```

Video object segmentation:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval_vos_ms.py \
  --dataset D17 \
  --weights models/cosine/mscosine_vos_fd1_md6_lr1e-4_ep50/pytorch_model/pytorch_model_24ep/pytorch_model.bin \
  --output outputs/repro/vos/d17 \
  --num_frame 6 \
  --memory_decay_type linear \
  --memory_decay_ratio 20 \
  --fix_first_frame
```

YouTube-VOS 2019 uses the same canonical VOS checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval_vos_ms.py \
  --dataset Y19 \
  --weights models/cosine/mscosine_vos_fd1_md6_lr1e-4_ep50/pytorch_model/pytorch_model_24ep/pytorch_model.bin \
  --output outputs/repro/vos/y19 \
  --num_frame 6 \
  --memory_decay_type linear \
  --memory_decay_ratio 20 \
  --fix_first_frame
```

Few-shot instance segmentation:

```bash
cd inference_fsod
python tools/preprocess.py \
  --config-file configs/COCO/1shots/seed0.yaml \
  --opts OUTPUT_DIR outputs/repro/fsod/coco/1shot/seed0 \
  MODEL.COSINE.preprocess True \
  MODEL.WEIGHTS ../models/cosine/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin

python tools/test.py \
  --num-gpus=1 \
  --config-file configs/COCO/1shots/seed0.yaml \
  --opts MODEL.COSINE.preprocess False \
  OUTPUT_DIR outputs/repro/fsod/coco/1shot/seed0 \
  MODEL.WEIGHTS ../models/cosine/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin
```

Minimal FSOD model-load smoke:

```bash
cd inference_fsod
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  --smoke-model-only \
  --config-file configs/LVIS/MSCOSINE_all_fc.yaml \
  --opts OUTPUT_DIR outputs/smoke/fsod_lvis \
  MODEL.WEIGHTS ../models/cosine/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin
```

## Verified Checkpoint Download

The public ModelScope checkpoint package was downloaded and verified on
2026-06-15. All four checkpoint files matched `SHA256SUMS.txt`, and
`scripts/check_required_assets.sh --weights-only` passed against the downloaded
layout.

Verified files:

| Checkpoint file | Size |
| --- | ---: |
| `weights/cosine_fd1_md6_lr1e-4_ep50/pytorch_model.bin` | 2,725,759,934 bytes |
| `weights/mscosine_refseg_fd1_md6_lr1e-4_ep50/pytorch_model_19ep/pytorch_model.bin` | 2,764,208,701 bytes |
| `weights/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin` | 2,764,208,701 bytes |
| `weights/mscosine_vos_fd1_md6_lr1e-4_ep50/pytorch_model/pytorch_model_24ep/pytorch_model.bin` | 2,764,208,701 bytes |

## Verified Minimal Smoke

The following bounded run was verified on the lab NAS with the public
checkpoint names. This is a functional smoke check, not a full paper metric
reproduction.

```bash
bash scripts/check_required_assets.sh

CUDA_VISIBLE_DEVICES=0 python tools/eval_fss_ms.py \
  --benchmark coco --fold 0 --nshot 1 --max-samples 1 \
  --log-root outputs/smoke/fss_ms_coco_fold0 \
  --use_text --use_visual --nworker 0

CUDA_VISIBLE_DEVICES=0 python tools/eval_referseg_ms.py \
  --val_dataset "refcoco|unc|val" --max-samples 1 \
  --output_dir outputs/smoke/refseg_refcoco \
  --num_workers 0 --device cuda

CUDA_VISIBLE_DEVICES=0 python tools/eval_vos_ms.py \
  --dataset D17 --max-videos 1 --max-frames 2 \
  --output outputs/smoke/vos_d17 \
  --num_frame 2 --fix_first_frame \
  --size 480 --img-size 896 --pad-size 896 --clip_image_size 1024

cd inference_fsod
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  --smoke-model-only \
  --config-file configs/LVIS/MSCOSINE_all_fc.yaml \
  --opts OUTPUT_DIR outputs/smoke/fsod_lvis \
  MODEL.WEIGHTS ../models/cosine/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin
```

Observed smoke outputs:

| Task | Bounded setting | Observed output |
| --- | --- | --- |
| FSS | COCO fold0, 1-shot, 1 batch | Completed; `FB-IoU 41.03` |
| RefSeg | `refcoco|unc|val`, 1 sample | `giou 0.8987`, `ciou 0.8987` |
| VOS | DAVIS 2017, 1 video / 2 frames | Wrote two masks under `outputs/smoke/vos_d17/bike-packing/` |
| FSOD | LVIS model-build/load check | `FSOD smoke-model-only succeeded.` |

The same public checkpoint layout was re-checked after the ModelScope public
package was published on 2026-06-15. The run used
`zzzmmz/COSINE-Public-Weights` checkpoint names and all steps exited with
`rc=0`:

| Task | Bounded setting | Observed output |
| --- | --- | --- |
| Asset check | Public checkpoint layout | All four COSINE checkpoints found |
| FSS | COCO fold0, 1-shot, 1 batch | Completed; `FB-IoU 41.03` |
| RefSeg | `refcoco|unc|val`, 1 sample | `giou 0.8987`, `ciou 0.8987` |
| VOS | DAVIS 2017, 1 video / 2 frames | Wrote two masks under `outputs/smoke_public_*/vos_d17/bike-packing/` |
| FSOD | LVIS model-build/load check | `FSOD smoke-model-only succeeded.` |

## Verified Full Runs

The following unbounded evaluation was verified on the lab NAS on 2026-06-14.
It is the template full-run command for the remaining task reproductions.

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/eval_fss_ms.py \
  --datapath datasets/fss \
  --benchmark coco \
  --fold 0 \
  --nshot 1 \
  --weights models/cosine/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin \
  --log-root outputs/repro/fss_full/mscosine_unified_fd1_md6_lr1e-4_ep50_49ep/coco/1shot/fold0_visual_text \
  --use_text --use_visual \
  --nworker 0
```

Observed output:

```text
COCO-20i multi-scale fold0, 1-shot, visual+text
Fold 0 mIoU: 73.53
FB-IoU: 84.83
Elapsed wall time: 15:09.69
Exit status: 0
Log: outputs/repro/fss_full/mscosine_unified_fd1_md6_lr1e-4_ep50_49ep/coco/1shot/fold0_visual_text/run.log
```

Additional representative full evaluations verified on the lab NAS:

| Task | Setting | Checkpoint | Launcher | Log | Reproduced |
| --- | --- | --- | --- | --- | --- |
| FSS COCO-20i | multi-scale / 1-shot / fold0 / visual+text | `models/cosine/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin` | `outputs/repro/fss_full/mscosine_unified_fd1_md6_lr1e-4_ep50_49ep/coco/1shot/fold0_visual_text/run.sh` | `outputs/repro/fss_full/mscosine_unified_fd1_md6_lr1e-4_ep50_49ep/coco/1shot/fold0_visual_text/run.log` | `mIoU 73.53`, `FB-IoU 84.83` |
| FSS COCO-20i | single-scale / 1-shot / fold0 / visual+text | `models/cosine/cosine_fd1_md6_lr1e-4_ep50/pytorch_model.bin` | `scripts/repro/run_representative_full_suite.sh` | `outputs/repro/fss_full/cosine_fd1_md6_lr1e-4_ep50/coco/1shot/fold0_visual_text/run.log` | `mIoU 71.61`, `FB-IoU 83.97` |
| RefCOCO | val | `models/cosine/mscosine_refseg_fd1_md6_lr1e-4_ep50/pytorch_model_19ep/pytorch_model.bin` | `scripts/repro/run_representative_full_suite.sh` | `outputs/repro/referseg_full/mscosine_refseg_fd1_md6_lr1e-4_ep50/refcoco_val/run.log` | `giou 0.7844`, `ciou 0.7724` |
| DAVIS17 | val / 24ep / official evaluator | `models/cosine/mscosine_vos_fd1_md6_lr1e-4_ep50/pytorch_model/pytorch_model_24ep/pytorch_model.bin` | `scripts/repro/run_representative_full_suite.sh` | prediction: `outputs/repro/vos_full/mscosine_vos_fd1_md6_lr1e-4_ep50/24ep/D17/run.log`; eval: `outputs/repro/vos_full/mscosine_vos_fd1_md6_lr1e-4_ep50/24ep/D17_eval/run.log` | `J&F 0.802`, `J 0.770`, `F 0.835` |
| YouTube-VOS 2019 | val / 24ep | `models/cosine/mscosine_vos_fd1_md6_lr1e-4_ep50/pytorch_model/pytorch_model_24ep/pytorch_model.bin` | `scripts/repro/run_representative_full_suite.sh` | `outputs/repro/vos_full/mscosine_vos_fd1_md6_lr1e-4_ep50/24ep/Y19/run.log` | Generated predictions for `507/507` videos |
| LVIS FSOD | `MSCOSINE_all_fc` / `vis_text` | `models/cosine/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin` | fresh cache + inference: `scripts/repro/run_fsod_lvis_fresh_full.sh`; JSON metric replay: `scripts/repro/eval_fsod_lvis_from_json.py` | inference: `outputs/repro/fsod_full/MSCOSINE_all_fc/vis_text/preprocess_and_test.log`; metrics: `outputs/repro/fsod_full/MSCOSINE_all_fc/vis_text/eval_from_json_patch.log` | `bbox AP 19.75`, `segm AP 20.28` |
| LVIS-92i | multi-scale / 1-shot / fold0 / visual+text | `models/cosine/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin` | `scripts/repro/run_lvis_fss_representative.sh` | `outputs/repro/fss_full/mscosine_unified_fd1_md6_lr1e-4_ep50_49ep/lvis/1shot/fold0_visual_text/run.log` | `mIoU 41.30`, `FB-IoU 70.01` |
| LVIS-92i | single-scale / 1-shot / fold0 / visual+text | `models/cosine/cosine_fd1_md6_lr1e-4_ep50/pytorch_model.bin` | `scripts/repro/run_lvis_fss_representative.sh` | `outputs/repro/fss_full/cosine_fd1_md6_lr1e-4_ep50/lvis/1shot/fold0_visual_text/run.log` | `mIoU 40.65`, `FB-IoU 70.26` |

FSOD JSON metric replay was launched from `inference_fsod/` with:

```bash
PYTHONPATH=. python ../scripts/repro/eval_fsod_lvis_from_json.py \
  2>&1 | tee ../outputs/repro/fsod_full/MSCOSINE_all_fc/vis_text/eval_from_json_patch.log
```

## Result Table

| Benchmark | Setting | Script | Checkpoint | Expected | Reproduced | Status |
| --- | --- | --- | --- | ---: | ---: | --- |
| COCO-20i | multi-scale / 1-shot / fold0 | `tools/eval_fss_ms.py` | `mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep` | TBD | Full fold0: `mIoU 73.53`, `FB-IoU 84.83` | Full fold0 passed |
| COCO-20i | single-scale / 1-shot / fold0 | `tools/eval_fss.py` | `cosine_fd1_md6_lr1e-4_ep50` | TBD | Full fold0: `mIoU 71.61`, `FB-IoU 83.97` | Full fold0 passed |
| LVIS-92i | multi-scale / 1-shot / fold0 | `tools/eval_fss_ms.py` | `mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep` | TBD | Full fold0: `mIoU 41.30`, `FB-IoU 70.01` | Full fold0 passed |
| LVIS-92i | single-scale / 1-shot / fold0 | `tools/eval_fss.py` | `cosine_fd1_md6_lr1e-4_ep50` | TBD | Full fold0: `mIoU 40.65`, `FB-IoU 70.26` | Full fold0 passed |
| RefCOCO | val | `tools/eval_referseg_ms.py` | `mscosine_refseg_fd1_md6_lr1e-4_ep50/pytorch_model_19ep` | TBD | Full val: `giou 0.7844`, `ciou 0.7724` | Full val passed |
| DAVIS17 | val | `tools/eval_vos_ms.py` plus official evaluator | `mscosine_vos_fd1_md6_lr1e-4_ep50/pytorch_model_24ep` | TBD | Full val: `J&F 0.802`, `J 0.770`, `F 0.835` | Full val passed |
| YouTube-VOS 2019 | val | `tools/eval_vos_ms.py --dataset Y19` | `mscosine_vos_fd1_md6_lr1e-4_ep50/pytorch_model_24ep` | TBD | Predictions generated for `507/507` videos | Predictions generated |
| LVIS FSOD | `MSCOSINE_all_fc` / `vis_text` | `inference_fsod/scripts/lvis_ms_fcclip.sh` | `mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep` | TBD | Full val: `bbox AP 19.75`, `segm AP 20.28` | Full eval passed |

## Release Criteria

- No private absolute paths remain in tracked code.
- No dataset, checkpoint, or generated output blobs are tracked.
- Public checkpoint links are added to `MODEL_ZOO.md`.
- At least one smoke command per task is run and recorded above.
