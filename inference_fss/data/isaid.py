r""" iSAID-5i few-shot semantic segmentation dataset """
import os

import torch
import PIL.Image as Image
import numpy as np
import torchvision.transforms as transforms2
from .pascal import DatasetPASCAL


class DatasetISAID(DatasetPASCAL):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize, aug=False) -> None:
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 3
        self.nclass = 15
        self.benchmark = 'isaid'
        self.base_path = os.path.join(datapath, 'iSAID')
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize

        datapath = os.path.join(datapath, 'iSAID/iSAID_patches')

        if self.split == 'trn':
            self.img_path = os.path.join(datapath, 'train/images')
            self.ann_path = os.path.join(datapath, 'train/semantic_png')
        else:
            self.img_path = os.path.join(datapath, 'val/images')
            self.ann_path = os.path.join(datapath, 'val/semantic_png')

        self.aug = aug and (self.split == 'trn')
        if self.aug:
            self.tv2 = transforms2.Compose([
                transforms2.RandomHorizontalFlip(),
                transforms2.RandomRotation(30),
                # transforms2.RandomResizedCrop(size=256, scale=(0.5, 1.0))
            ])

        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return len(self.img_metadata)  # TODO: why hsnet use 100 for val

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        # mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '_instance_color_RGB.png')))
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '_instance_color_RGB.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.png')

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join(self.base_path, 'splits/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from matcher.common import utils
    from tqdm import tqdm
    utils.fix_randseed(0)

    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    from torchvision import transforms
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    datapath = 'datasets'
    use_original_imgsize = False
    fold = 0
    split = 'val'

    # cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize(cls.img_mean, cls.img_std)])

    transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                        transforms.ToTensor()])
    for fold in tqdm([0,1,2]):

        dataset = DatasetISAID(datapath, fold=fold, transform=transform, split=split, shot=1,
                                          use_original_imgsize=False)

        for idx in range(len(dataset)):

            if idx > 20:
                break

            batch = dataset[idx]

            query_img, query_mask, support_imgs, support_masks = \
                batch['query_img'], batch['query_mask'], \
                batch['support_imgs'], batch['support_masks']

            imgs = torch.cat([query_img, support_imgs.squeeze()], dim=-1).permute(1,2,0).numpy()
            masks = torch.cat([query_mask, support_masks.squeeze()], dim=-1).numpy()

            query_n = batch['query_name'].split('/')[-1]

            if not os.path.exists(f'shows/isaid/fold{fold}'):
                os.makedirs(f'shows/isaid/fold{fold}')

            save_path = f'shows/isaid/fold{fold}/{query_n}'
            plt.figure(figsize=(10, 10))
            plt.imshow(imgs)
            show_mask(masks[None, ...], plt.gca())
            plt.axis('off')
            plt.savefig(save_path)
