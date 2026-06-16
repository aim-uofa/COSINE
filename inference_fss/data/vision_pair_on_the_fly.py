r""" FSS-1000 few-shot semantic segmentation dataset """
import os
import glob
import random

from collections import defaultdict
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


class DatasetVISION(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, num=600, use_original_imgsize=False):
        self.split = split
        self.benchmark = 'vision'
        self.shot = shot
        self.num = num
        self.nfolds = 1
        self.nclass = 2

        self.base_path = os.path.join(datapath, 'test-data-for-ZJU')
        self.base_name = os.path.basename(self.base_path)

        # Given predefined test split, load randomly generated training/val splits:
        # (reference regarding trn/val/test splits: https://github.com/HKUSTCV/FSS-1000/issues/7))
        sub_cat_dirs = glob.glob(self.base_path + "/*/*")
        self.categories = [x.split(self.base_name)[-1][1:] for x in sub_cat_dirs]
        self.categories = sorted(self.categories)
        assert len(self.categories) == self.nclass

        # example of self.categories: ['Cable/thunderbolt', 'Screw/front']

        self.class_ids = self.build_class_ids()
        self.img_metadata, self.sub_cat_to_img_mapping = self.build_img_metadata()

        self.transform = transform

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,

                 'class_id': torch.tensor(class_sample)}

        return batch

    def load_frame(self, query_name, support_names):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        query_mask_name = query_name.replace("jpg","png").replace("image","label")
        support_mask_names = [name.replace("jpg","png").replace("image","label") for name in support_names]

        query_mask = self.read_mask(query_mask_name)
        support_masks = [self.read_mask(name) for name in support_mask_names]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        class_id = idx % len(self.class_ids)
        class_sample = self.categories[class_id]
        # example of query_name: 'datasets/240530-for-data-challenge/train/Cable/thunderbolt/image/000000.jpg'

        query_name = np.random.choice(self.sub_cat_to_img_mapping[class_sample], 1, replace=False)[0]
        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.sub_cat_to_img_mapping[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_id

    def build_class_ids(self):
        if self.split == 'test':
            class_ids = range(0, len(self.categories))
        else:
            raise ValueError(f"Split {self.split} is not supported for VISION Dataset")
        return class_ids

    def build_img_metadata(self):
        img_metadata = []
        sub_cat_to_img_mapping = defaultdict(list)
        for sub_cat in self.categories:
            sub_cat_dir = os.path.join(self.base_path, sub_cat)
            img_paths = sorted(glob.glob(sub_cat_dir + "/image/*.jpg"))
            for img_path in img_paths:
                img_metadata.append(img_path)
                sub_cat_to_img_mapping[sub_cat].append(img_path)
        return img_metadata, sub_cat_to_img_mapping

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    # from ..common import utils
    from tqdm import tqdm
    # utils.fix_randseed(0)

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
    datapath = "datasets"
    use_original_imgsize = False
    fold = 0
    split = 'test'

    # cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize(cls.img_mean, cls.img_std)])

    transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                        transforms.ToTensor()])
    for fold in tqdm([0]):

        dataset = DatasetVISION(datapath, fold=fold, transform=transform, split=split, shot=1,
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

            if not os.path.exists(f'shows/vison/fold{fold}'):
                os.makedirs(f'shows/vison/fold{fold}')

            save_path = f'shows/vison/fold{fold}/{query_n}'
            plt.figure(figsize=(10, 10))
            plt.imshow(imgs)
            show_mask(masks[None, ...], plt.gca())
            plt.axis('off')
            plt.savefig(save_path)
