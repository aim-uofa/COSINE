import torch
from torchvision import transforms
import numpy as np

from detectron2.data import transforms as T
from dinov2.data.transforms import MaybeToTensor, make_normalize_transform

from cosine.data.coco import COCOPanoDataset
from cosine.data.coco_ins import COCOInsDataset
from cosine.data.paco import PACODataset
from cosine.data.o365 import O365Dataset
from cosine.data.refcoco import RefCOCODataset

class HybridDataset(torch.utils.data.Dataset):

    dino_transform = transforms.Compose([
        MaybeToTensor(),
        make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_format = 'RGB'
    ins_data_name_mapping_train = {
        'coco': 'coco_2017_train_panoptic',
        'ade20k': 'ade20k_panoptic_train',
        'lvis': 'lvis_v1_train_ins',
        'paco': 'paco_lvis_v1_train',
        'o365': 'object365_train',
    }

    ins_data_name_mapping_val = {
        'coco': 'coco_2017_val_panoptic',
        'ade20k': 'ade20k_panoptic_val',
        'lvis': 'lvis_v1_val_ins',
        'paco': 'paco_lvis_v1_val',
        'o365': 'object365_val'
    }

    refer_data_name_mapping_train = {
        'refclef': 'refclef',
        'refcoco': 'refcoco',
        'refcoco+': 'refcoco+',
        'refcocog': 'refcocog'
    }

    def __init__(
        self,
        image_size,
        sam_image_size,
        clip_image_size,
        root='datasets',
        crop_ratio=0.5,
        samples_per_epoch=500 * 8 * 2 * 10,
        datasets="ins_seg||refer_seg",
        ins_seg_data="coco||o365||paco",
        ins_sample_rate=[1, 1, 1],
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        refer_sample_rate=[1, 1, 1, 1],
        multimodal_choice=['visual', 'text', 'visual_text', 'refer'],
        use_all_classes=False,
        tfm_gens_crop_pair=None,
        tfm_gens_sel_pair=None,
        tfm_refer=None,
        is_train=True,
    ):
        DatasetDict = {
            'ins_seg':
                {
                'coco': COCOPanoDataset,
                'o365': O365Dataset,
                'lvis': COCOInsDataset,
                'paco': PACODataset,
                },
            'refer_seg':
                {
                'refclef': RefCOCODataset,
                'refcoco': RefCOCODataset,
                'refcoco+': RefCOCODataset,
                'refcocog': RefCOCODataset
            }
        }

        self.is_train = is_train
        self.ins_data_name_mapping = self.ins_data_name_mapping_train if is_train else self.ins_data_name_mapping_val
        self.refer_data_name_mapping = self.refer_data_name_mapping_train

        self.root_ = root
        self.image_size = image_size
        self.samples_per_epoch = samples_per_epoch

        self.datasets = datasets.split("||")
        self.seg_dataset_dict = {
            'ins_seg': ins_seg_data.split("||") if 'ins_seg' in datasets else None,
            'refer_seg': refer_seg_data.split("||") if 'refer_seg' in datasets else None,
        }

        ins_sample_rate = np.array(ins_sample_rate)
        refer_sample_rate = np.array(refer_sample_rate)

        self.dataset_sample_rate_dict = {
            'ins_seg': ins_sample_rate / ins_sample_rate.sum() if 'ins_seg' in datasets else None,
            'refer_seg': refer_sample_rate / refer_sample_rate.sum() if 'refer_seg' in datasets else None,
        }

        self.multimodal_choice = multimodal_choice

        self.all_datasets = {ds: [] for ds in self.datasets}
        for dataset in self.datasets:
            Dataset = DatasetDict[dataset]

            for seg_dataset in self.seg_dataset_dict[dataset]:
                from cosine.utils.utils import Print; Print(f"loading {seg_dataset} !")

                if 'ins' in dataset:
                    dataset_name = self.ins_data_name_mapping[seg_dataset]

                    self.all_datasets[dataset].append(
                        Dataset[seg_dataset](
                            image_size=image_size,
                            sam_image_size=sam_image_size,
                            clip_image_size=clip_image_size,
                            root=root,
                            dataset_name=dataset_name,
                            crop_ratio=crop_ratio,
                            tfm_gens_crop_pair=tfm_gens_crop_pair,
                            tfm_gens_sel_pair=tfm_gens_sel_pair,
                            dino_transform=self.dino_transform,
                            use_all_classes=use_all_classes,
                        )
                    )

                elif 'refer' in dataset:
                    dataset_name = self.refer_data_name_mapping[seg_dataset]

                    self.all_datasets[dataset].append(
                        Dataset[seg_dataset](
                            image_size=image_size,
                            sam_image_size=sam_image_size,
                            clip_image_size=clip_image_size,
                            root=root,
                            dataset_name=dataset_name,
                            refer_seg_data=seg_dataset,
                            transform=tfm_refer,
                            dino_transform=self.dino_transform,
                        )
                    )


    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        datas = {}

        if 'visual' in self.multimodal_choice:
            datasets = self.all_datasets['ins_seg']
            ind = np.random.choice(list(range(len(datasets))), p=self.dataset_sample_rate_dict['ins_seg'])
            dataset = datasets[ind]
            dataset.set_multimodal(visual_prompt=True, text_prompt=False)
            datas['visual'] = dataset[idx]
        if 'text' in self.multimodal_choice:
            datasets = self.all_datasets['ins_seg']
            ind = np.random.choice(list(range(len(datasets))), p=self.dataset_sample_rate_dict['ins_seg'])
            dataset = datasets[ind]
            dataset.set_multimodal(visual_prompt=False, text_prompt=True)
            datas['text'] = dataset[idx]
        if 'visual_text' in self.multimodal_choice:
            datasets = self.all_datasets['ins_seg']
            ind = np.random.choice(list(range(len(datasets))), p=self.dataset_sample_rate_dict['ins_seg'])
            dataset = datasets[ind]
            dataset.set_multimodal(visual_prompt=True, text_prompt=True)
            datas['visual_text'] = dataset[idx]
        if 'refer' in self.multimodal_choice:
            datasets = self.all_datasets['refer_seg']
            ind = np.random.choice(list(range(len(datasets))), p=self.dataset_sample_rate_dict['refer_seg'])
            dataset = datasets[ind]
            datas['refer'] = dataset[idx]

        return datas

def build_augmentation(args, is_train):
    augmentation = []

    if is_train:
        # LSJ aug
        if args.random_flip != "none":
            augmentation.append(
                T.RandomFlip(
                    horizontal=args.random_flip == "horizontal",
                    vertical=args.random_flip == "vertical",
                )
            )

        augmentation.extend([
            T.ResizeScale(
                min_scale=args.min_scale, max_scale=args.max_scale, target_height=args.image_size,
                target_width=args.image_size
            ),
            T.FixedSizeCrop(crop_size=(args.image_size, args.image_size), pad=False),
        ])

    else:
        augmentation.append(
            T.ResizeShortestEdge(
                short_edge_length=args.image_size,
                max_size=args.image_size
            )
        )

    return augmentation


def build_refer_augmentation(args, is_train):
    augmentation = []

    if is_train:

        augmentation.extend([
            T.ResizeShortestEdge(args.min_size, args.max_size, "choice")
        ])

    else:
        augmentation.append(
            T.ResizeShortestEdge(
                short_edge_length=args.image_size,
                max_size=args.image_size
            )
        )

    return augmentation

def build_dataset(args, is_train):

    assert is_train
    augmentation = build_augmentation(args, is_train)
    refer_augmentation = build_refer_augmentation(args, is_train)


    dataset = HybridDataset(
        image_size=args.image_size,
        sam_image_size=args.sam_image_size,
        clip_image_size=args.clip_image_size,
        root=args.data_root,
        crop_ratio=args.crop_ratio,
        samples_per_epoch=args.steps_per_epoch * args.world_size * args.batch_size * args.update_freq,
        datasets=args.dataset,
        ins_seg_data=args.ins_seg_data,
        ins_sample_rate=[float(x) for x in args.ins_sample_rate.split(",")],
        refer_seg_data=args.refer_seg_data,
        refer_sample_rate=[float(x) for x in args.refer_sample_rate.split(",")],
        multimodal_choice=[x for x in args.multimodal_choice.split("||")],
        use_all_classes=args.use_all_classes,
        tfm_gens_crop_pair=augmentation,
        tfm_gens_sel_pair=augmentation,
        tfm_refer=refer_augmentation,
        is_train=is_train
    )


    return dataset


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser('coco dataset', add_help=False)
    parser.add_argument('--data_root', default="datasets", type=str)
    parser.add_argument('--dataset', default="ins_seg||ref_seg", type=str)
    parser.add_argument('--ins_seg_data', default="coco||paco||o365", type=str)
    parser.add_argument('--ins_sample_rate', default="1,1,1", type=str)


    parser.add_argument('--random_flip', default="horizontal", type=str)
    parser.add_argument('--min_scale', default=0.1, type=float)
    parser.add_argument('--max_scale', default=2.0, type=float)
    parser.add_argument('--min_size', default=(560, 588, 616, 644, 672, 700), type=tuple)
    parser.add_argument('--max_size', default=896, type=int)
    parser.add_argument('--image_size', default=896, type=int)
    parser.add_argument('--sam_image_size', default=1024, type=int)
    parser.add_argument('--clip_image_size', default=1024, type=int)
    parser.add_argument('--use_all_classes', action='store_true')
    parser.set_defaults(use_all_classes=True)
    args = parser.parse_args()

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    is_train = True
    augmentation = build_augmentation(args, is_train)
    refer_aug = build_refer_augmentation(args, is_train)


    dataset = HybridDataset(
        is_train=is_train,
        image_size=args.image_size,
        sam_image_size = args.sam_image_size,
        clip_image_size = args.clip_image_size,
        root='datasets',
        crop_ratio=0.5,
        samples_per_epoch=5000 * 8 * 2 * 10,
        datasets="ins_seg||refer_seg",
        ins_seg_data="coco",
        ins_sample_rate=[1],
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        refer_sample_rate=[1, 1, 1, 1],
        multimodal_choice=['visual', 'text', 'visual_text', 'refer'],
        use_all_classes=True,
        tfm_gens_crop_pair=augmentation,
        tfm_gens_sel_pair=augmentation,
        tfm_refer=refer_aug
    )


    for id in range(len(dataset)):

        datas = dataset[id]
        print()
