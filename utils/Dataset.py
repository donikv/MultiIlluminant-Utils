import os
from math import floor, ceil

import albumentations as albu
import cv2
import torch
from albumentations import pytorch as AT
from torch.utils.data import Dataset
import numpy as np

from dataset_utils import load_img_and_gt, visualize, get_mask_from_gt, mask_to_image, get_patch_with_index


class MIDataset(Dataset):
    def __init__(self, path: str = './data', folder: str = 'dataset_crf/lab', special_folder: str = '',
                 df: object = None, datatype: object = 'train',
                 img_ids: object = None,
                 transforms: object = albu.Compose([albu.HorizontalFlip(), AT.ToTensor()]),
                 preprocessing: object = None,
                 use_mask: bool = False) -> object:
        self.df = df
        self.datatype = datatype
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.use_mask = use_mask
        self.path = path
        self.folder = folder

        if self.datatype != 'test':
            self.gt_names = os.listdir(f"{path}/{folder}/groundtruth/{special_folder}")

        self.image_names = os.listdir(f"{path}/{folder}/srgb8bit/{special_folder}")

        self.path = path
        self.folder = folder

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image, gt, mask = load_img_and_gt(image_name, self.path, self.folder, use_mask=self.use_mask)
        augmented = self.transforms(image=image, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        augmented = self.transforms(image=image, mask=gt)
        gt = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
            preprocessed = self.preprocessing(image=img, mask=gt)
            gt = preprocessed['mask']
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32, device="cuda")
        # mask = np.apply_along_axis(lambda x: x[], 2, mask)
        gt = torch.tensor(gt.transpose(2, 0, 1), dtype=torch.float32, device="cuda")
        mask = np.array([[(np.array([1, 0]) if pixel[0] > 128 else np.array([0, 1])) for pixel in row] for row in mask])
        # mask = get_mask_from_gt(gt)
        # name = f"{self.path}/{self.folder}/gt_mask/{image_name}"
        # print(mask)
        # print(name)
        # cv2.imwrite(name, mask_to_image(mask))
        mask = torch.tensor(mask.transpose(2, 0, 1), dtype=torch.float32, device="cuda")
        return img, \
               mask, \
               gt

    def __len__(self):
        return len(self.image_names)


class MIPatchedDataset(MIDataset):
    def __init__(self, path: str = './data', folder: str = 'dataset_crf/lab', special_folder: str = '',
                 df: object = None, datatype: object = 'train',
                 img_ids: object = None,
                 transforms: object = albu.Compose([albu.HorizontalFlip(), AT.ToTensor()]),
                 preprocessing: object = None,
                 use_mask: bool = False,
                 patch_width_ratio: float = 1. / 10, patch_height_ratio: float = 1. / 10) -> object:
        super(MIPatchedDataset, self).__init__(path, folder, special_folder, df, datatype, img_ids, transforms,
                                               preprocessing, use_mask)
        self.patch_width_ratio = patch_width_ratio
        self.patch_height_ratio = patch_height_ratio

    def __getitem__(self, idx):
        patches_per_img = ceil(1 / (self.patch_height_ratio * self.patch_width_ratio))
        image_idx = int(idx / patches_per_img)
        image_name = self.image_names[image_idx]
        image, gt, mask = load_img_and_gt(image_name, self.path, self.folder, use_mask=self.use_mask)

        image = get_patch_with_index(image, image_idx, self.patch_height_ratio, self.patch_width_ratio)
        mask = get_patch_with_index(mask, image_idx, self.patch_height_ratio, self.patch_width_ratio)
        gt = get_patch_with_index(gt, image_idx, self.patch_height_ratio, self.patch_width_ratio)

        augmented = self.transforms(image=image, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        augmented = self.transforms(image=image, mask=gt)
        gt = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
            preprocessed = self.preprocessing(image=img, mask=gt)
            gt = preprocessed['mask']
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32, device="cuda")

        gt = torch.tensor(gt.transpose(2, 0, 1), dtype=torch.float32, device="cuda")
        mask = np.array([[(np.array([1, 0]) if pixel[0] > 128 else np.array([0, 1])) for pixel in row] for row in mask])
        mask = torch.tensor(mask.transpose(2, 0, 1), dtype=torch.float32, device="cuda")
        return img, \
               mask, \
               gt

    def __len__(self):
        return floor(len(self.image_names) / self.patch_width_ratio / self.patch_height_ratio)
