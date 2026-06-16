r""" ISIC few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import pandas as pd


class DatasetISIC(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, num=600, use_original_imgsize=False):
        self.split = split
        self.benchmark = 'isic'
        self.shot = shot
        self.num = num
        self.nfolds = 1
        self.nclass = 3

        self.base_path = os.path.join(datapath, 'ISIC')
        self.categories = ['1','2','3']

        self.class_ids = range(0, 3)
        self.cls_dict = {'nevus': "1", 'melanoma': "2", 'seborrheic_keratosis': "3"}
        self.img_metadata_classwise = self.build_img_metadata_classwise()

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

        query_id = query_name.split('/')[-1].split('.')[0]
        ann_path = os.path.join(self.base_path, 'ISIC2018_Task1_Training_GroundTruth')
        query_name = os.path.join(ann_path, query_id) + '_segmentation.png'
        support_ids = [name.split('/')[-1].split('.')[0] for name in support_names]
        support_names = [os.path.join(ann_path, sid) + '_segmentation.png' for name, sid in zip(support_names, support_ids)]

        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        class_id = idx % len(self.class_ids)
        class_sample = self.categories[class_id]

        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_id

    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:
            os.path.join(self.base_path, cat)
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, 'ISIC2018_Task1-2_Training_Input', cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg':
                    img_metadata.append(img_path)
        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for cat in self.categories:
            img_metadata_classwise[cat] = []

        class_ids = pd.read_csv(f"{self.base_path}/class_id.csv")
        # print(class_ids)
        # print(class_ids.loc[class_ids["ID"] == "ISIC_0000000", "Class"][0])
        # print(os.path.join(self.base_path, 'ISIC2018_Task1-2_Training_Input'))

        # for cat in self.categories:
            # img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, 'ISIC2018_Task1-2_Training_Input', cat))])
            # for img_path in img_paths:
            #     if os.path.basename(img_path).split('.')[1] == 'jpg':
            #         img_metadata_classwise[cat] += [img_path]
        img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, 'ISIC2018_Task1-2_Training_Input'))])
        for img_path in img_paths:
            if os.path.basename(img_path).split('.')[1] == 'jpg':
                # print(class_ids.loc[class_ids["ID"] == os.path.basename(img_path).split('.')[0], "Class"].values[0])
                img_metadata_classwise[self.cls_dict[class_ids.loc[class_ids["ID"] == os.path.basename(img_path).split('.')[0], "Class"].values[0]]] += [img_path]
        return img_metadata_classwise


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
    for fold in tqdm([0,]):

        dataset = DatasetISIC(datapath, fold=fold, transform=transform, split=split, shot=1,
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

            if not os.path.exists(f'shows/isic/fold{fold}'):
                os.makedirs(f'shows/isic/fold{fold}')

            save_path = f'shows/isic/fold{fold}/{query_n}'
            plt.figure(figsize=(10, 10))
            plt.imshow(imgs)
            show_mask(masks[None, ...], plt.gca())
            plt.axis('off')
            plt.savefig(save_path)