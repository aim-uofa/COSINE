r""" Chest X-ray few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


class DatasetLung(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, num=600, use_original_imgsize=False):
        self.split = split
        self.benchmark = 'lung'
        self.shot = shot
        self.num = num
        self.nfolds = 1
        self.nclass = 1

        self.base_path = os.path.join(datapath, 'LungSegmentation')
        self.img_path = os.path.join(self.base_path, 'CXR_png')
        self.ann_path = os.path.join(self.base_path, 'masks')

        self.categories = ['1']

        self.class_ids = range(0, 1)
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
        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

        query_id = query_name.split('/')[-1]
        if '_mask' in query_id:
            query_id = query_id.replace('_mask.png','.png')
        query_img = Image.open(os.path.join(self.img_path, os.path.basename(query_id))).convert('RGB')

        support_ids = [] # = [os.path.basename(name)[:-9] + '.png' for name in support_names]
        for name in support_names:
            sup_id = name.split('/')[-1]
            if '_mask' in sup_id:
                sup_id = sup_id.replace('_mask.png', '.png')
            support_ids.append(sup_id)
        support_names = [os.path.join(self.img_path, sid) for sid in support_ids]
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

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
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.img_path, cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'png':
                    img_metadata.append(img_path)
        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for cat in self.categories:
            img_metadata_classwise[cat] = []

        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % self.ann_path)])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'png':
                    img_metadata_classwise[cat] += [img_path]
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

        dataset = DatasetLung(datapath, fold=fold, transform=transform, split=split, shot=1,
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

            if not os.path.exists(f'shows/lung/fold{fold}'):
                os.makedirs(f'shows/lung/fold{fold}')

            save_path = f'shows/lung/fold{fold}/{query_n}'
            plt.figure(figsize=(10, 10))
            plt.imshow(imgs)
            show_mask(masks[None, ...], plt.gca())
            plt.axis('off')
            plt.savefig(save_path)
