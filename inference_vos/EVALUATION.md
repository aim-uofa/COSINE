## Evaluation - Video Object Segmentation


### Prepare data

Download datasets:

> python -m scripts.download_datasets

Create a directory 'datasets' for the above datasets and appropriately place each dataset to have following directory structure:

    datasets/
    ├── vos/
    │   ├── DAVIS16/
    │   │   ├── Annotations
    │   │   └── ...
    │   ├── DAVIS17
    │   │   ├── test-dev
    │   │   │   ├── Annotations
    │   │   │   └── ...
    │   │   └── trainval
    │   │       ├── Annotations
    │   └── YouTubeVOS18
    │       ├── all_frames
    │       │   └── valid_all_frames
    │       └── valid


### Testing


```
python tools/eval_vos.py \
  --dataset D17 \
  --weights /path/to/cosine_checkpoint/pytorch_model.bin \
  --output /path/vos/log \
  --img-size 1036 \
  --pad-size 1036 \
  --num_frame 6 \
  --memory_decay_type linear \
  --memory_decay_ratio 20 \
  --fix_first_frame

python inference_vos/davis2017-evaluation/evaluation_method.py \
  --davis_path datasets/vos/DAVIS17/trainval \
  --set val \
  --task semi-supervised \
  --results_path /path/vos/log
```