## Evaluation - Few-shot Semnatic Segmentation


### Prepare FSS Benchmark

Download following datasets:


> #### 1. COCO-20<sup>i</sup>
> Download COCO2014 train/val images and annotations:
> ```bash
> wget http://images.cocodataset.org/zips/train2014.zip
> wget http://images.cocodataset.org/zips/val2014.zip
> wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
> ```
> Download COCO2014 train/val annotations from this Google Drive: [train2014.zip](https://drive.google.com/file/d/1cwup51kcr4m7v9jO14ArpxKMA4O3-Uge/view?usp=sharing), [val2014.zip](https://drive.google.com/file/d/1PNw4U3T2MhzAEBWGGgceXvYU3cZ7mJL1/view?usp=sharing). (and locate both train2014/ and val2014/ under annotations/ directory).
> Download data [splits](https://github.com/juhongm999/hsnet/tree/main/data/splits/coco).


> #### 2. PASCAL-5<sup>i</sup>
> Download PASCAL VOC2012 devkit (train/val data):
> ```bash
> wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
> ```
> Download PASCAL VOC2012 SDS extended mask annotations from this [Google Drive](https://drive.google.com/file/d/10zxG2VExoEZUeyQl_uXga2OWHjGeZaf2/view?usp=sharing).
> Download data [splits](https://github.com/juhongm999/hsnet/tree/main/data/splits/pascal).

> #### 3. LVIS-92<sup>i</sup>
> Download COCO2017 train/val images:
> ```bash
> wget http://images.cocodataset.org/zips/train2017.zip
> wget http://images.cocodataset.org/zips/val2017.zip
> ```
> Download LVIS-92<sup>i</sup> extended mask annotations from our Google Drive: [lvis.zip](https://drive.google.com/file/d/1itJC119ikrZyjHB9yienUPD0iqV12_9y/view?usp=sharing).


Create a directory 'datasets' for the above datasets and appropriately place each dataset to have following directory structure:

    datasets/
    ├── fss/
        ├── COCO2014/
        │   ├── annotations/
        │   │   ├── train2014/
        │   │   └── val2014/
        │   ├── train2014/
        │   ├── val2014/
        │   └── splits
        │       ├── trn/
        │       └── val/
        ├── VOC2012/
        │   ├── Annotations/
        │   ├── ImageSets/
        │   ├── JPEGImages/
        │   ├── SegmentationClass/
        │   ├── SegmentationClassAug/
        │   ├── SegmentationObject/
        │   └── splits
        │       ├── trn/
        │       └── val/
        ├── LVIS/
        │   ├── coco/
        │   │   ├── train2017/
        │   │   └── val2017/
        │   ├── lvis_train.pkl
        │   └── lvis_val.pkl



### Testing

```

CUDA_VISIBLE_DEVICES=0 python tools/eval_fss.py  \
  --benchmark coco \
  --fold 0 \
  --nshot 1 \
  --weights /path/to/cosine_checkpoint/pytorch_model.bin \
  --log-root /path/to/fss/coco/1shot/fold0
```
