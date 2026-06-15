import torch
from torchvision import transforms
import numpy as np

from detectron2.data import transforms as T
from dinov2.data.transforms import MaybeToTensor, make_normalize_transform

from cosine.data.dataset_new import HybridDataset, build_refer_augmentation, build_augmentation

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


    def trivial_batch_collator(batch):
        """
        A batch collator that does nothing.
        """

        new_batch = {}
        for b in batch:
            for k in b:
                if k not in new_batch:
                    new_batch[k] = []
                new_batch[k].append(b[k])

        return new_batch

    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader_train = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=2,
        collate_fn=trivial_batch_collator
    )


    for batch in data_loader_train:

        print()