## Training COSINE

### Requirements
- Linux or macOS with Python ≥ 3.9
- PyTorch ≥ 2.0.1 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- `pip install -r requirements.txt`


### Example conda environment setup
```bash
conda create --name cosine python=3.9.17
conda activate cosine

pip install torch==2.0.1 torchvision==0.15.2
# or install xformers for faster inference of DINOv2
# pip install xformers==0.0.21 torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu117

git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

git clone https://github.com/aim-uofa/COSINE.git
cd COSINE
pip install -r requirements.txt
```

### Prepare pre-trained model

Download the model weights of [DINOv2](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth), and organize it as follows.
```
models/
    dinov2_vitl14_pretrain.pth
```


### Prepare training data

Download following datasets:


> #### COCO & ADE20K
> - Please prepare COCO & ADE20K datasets by referring to [Mask2Former](https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md).
> #### Objects365
> - Please refer [here](https://github.com/KainingYing/SAM_Objects365/tree/main?tab=readme-ov-file) to prepare SAM-Objects365 dataset. (This work is performed by Kaining Ying when he visits [AIM Lab](https://github.com/aim-uofa).)
> #### PACO
> - Download PACO-LVIS annotations from [here](https://dl.fbaipublicfiles.com/paco/annotations/paco_lvis_v1.zip).


Create a directory 'datasets' for the above datasets and appropriately place each dataset to have following directory structure:

    datasets/
    ├── coco/
    │   ├── annotations/
    │   │   ├── instances_{train,val}2017.json
    │   │   └── panoptic_{train,val}2017.json
    │   ├── {train,val}2017/
    │   ├── panoptic_{train,val}2017/
    │   └── panoptic_semseg_{train,val}2017/
    ├── ADEChallengeData2016/
    │   ├── images/
    │   ├── annotations/
    │   ├── objectInfo150.txt
    │   ├── annotations_instance/
    │   ├── annotations_detectron2/
    │   ├── ade20k_panoptic_{train,val}.json
    │   ├── ade20k_panoptic_{train,val}/
    │   ├── ade20k_instance_{train,val}.json
    ├── Objects365/
    │   ├── annotations
    │   │   ├── sam_obj365_train_1742k.json
    │   │   ├── sam_obj365_train_75k.json
    │   │   ├── sam_obj365_val_5k.json
    │   │   ├── zhiyuan_objv2_train.json
    │   │   └── zhiyuan_objv2_val.json
    │   ├── sam_mask_json
    │   │   ├── sam_obj365_train_1742k
    │   │   └── sam_obj365_train_75k
    │   ├── images
    │   │   ├── patch0
    │   │   ├── patch1
    │   │   └── ...
    ├── paco/
    │   ├── paco_lvis_v1_train.json
    │   ├── paco_lvis_v1_test.json
    │   └── paco_lvis_v1_val.json


### Training


```
python tools/train.py \
    --num-gpus 8 \
    --num-machines ${WORLD_SIZE} \
    --machine-rank ${RANK} \
    --dist-url ${DIST_URL} \
    --batch_size 2 \
    --epochs 50 \
    --update_freq 4 \
    --dinov2-size vit_large \
    --dinov2-weights models/dinov2_vitl14_pretrain.pth \
    --save_ckpt_freq 1 \
    --dataset "pano_seg||ins_seg" \
    --sample_rate "2,5" \
    --pano_seg_data "coco||ade20k" \
    --pano_sample_rate "1,1" \
    --ins_seg_data "coco||o365" \
    --ins_sample_rate "1,3" \
    --lr 1e-4 \
    --transformer_num_queries 200 \
    --transformer_fusion_layer_depth 1 \
    --crop_ratio 0.5 \
    --load_dir "" \
    --output_dir outputs/train/cosine_fd1_md6_lr1e-4_ep50 \
    --log_dir outputs/train/cosine_fd1_md6_lr1e-4_ep50 \
    --num_workers 2 \
    --auto_resume
```

When training is finished, to get the full model weight:

```
cd outputs/train/cosine_fd1_md6_lr1e-4_ep50 && python zero_to_fp32.py . pytorch_model.bin
```
