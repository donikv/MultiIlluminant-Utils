import os
from math import floor, ceil

import albumentations as albu
import cv2
import torch
from albumentations import pytorch as AT
from torch.utils.data import Dataset
import numpy as np

from dataset_utils import load_img_and_gt_crf_dataset, visualize, get_mask_from_gt, mask_to_image, get_patch_with_index, \
    load_img_hdr_dataset
from load_mat import get_gt_for_image
from transformation_utils import transform_to_log


class HDRDataset(Dataset):
    def __init__(self, images_path: str = './data/dataset_hdr/HDR/cs/chroma/data/Nikon_D700/HDR_MATLAB_3x3/',
                 gt_path: str = './data/dataset_hdr/real_illum/real_illum',
                 df: object = None, datatype: object = 'train',
                 transforms: object = albu.Compose([albu.HorizontalFlip(), AT.ToTensor()]),
                 preprocessing: object = None,
                 log_transform=False) -> object:
        self.df = df
        self.datatype = datatype
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.images_path = images_path
        self.gt_path = gt_path
        self.log_transform = log_transform

        all_images = os.listdir(self.images_path)
        self.gt_names = os.listdir(self.gt_path)
        self.gt_names = list(map(lambda x: x[:-4], self.gt_names))
        self.gt_names = list(filter(lambda x: x[:-11] + '.hdr' in all_images, self.gt_names))
        self.image_names = list(map(lambda x: x[:-11] + '.hdr', self.gt_names))

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        gt_name = self.gt_names[idx]
        image = load_img_hdr_dataset(image_name, self.images_path)
        gt = get_gt_for_image(gt_name, self.gt_path)
        gs, gt_gs = [], []
        if self.log_transform:
            image, gs = transform_to_log(image)
            gt, gt_gs = transform_to_log(gt)

        augmented = self.transforms(image=image, mask=gs)
        img, gs = augmented['image'], augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=gs)
            img, gs = preprocessed['image'], preprocessed['mask']
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32, device="cuda")
        gt = torch.tensor(gt, dtype=torch.float32, device='cuda')
        return img, gt, gs, gt_gs

    def __len__(self):
        return len(self.image_names)


class HDRPatchedDataset(HDRDataset):
    def __init__(self, images_path: str = './data/dataset_hdr/HDR/cs/chroma/data/Nikon_D700/HDR_MATLAB_3x3/',
                 gt_path: str = './data/dataset_hdr/real_illum/real_illum', df: object = None,
                 datatype: object = 'train', transforms: object = albu.Compose([albu.HorizontalFlip(), AT.ToTensor()]),
                 preprocessing: object = None, log_transform=False, patch_width_ratio: float = 1. / 10,
                 patch_height_ratio: float = 1. / 10) -> object:
        super().__init__(images_path, gt_path, df, datatype, transforms, preprocessing, log_transform)
        self.patch_width_ratio = patch_width_ratio
        self.patch_height_ratio = patch_height_ratio

    def __getitem__(self, idx):
        patches_per_img = ceil(1 / (self.patch_height_ratio * self.patch_width_ratio))
        image_idx = int(idx / patches_per_img)
        patch_idx = idx % patches_per_img

        image_name = self.image_names[image_idx]
        gt_name = self.gt_names[image_idx]

        image = load_img_hdr_dataset(image_name, self.images_path)
        gt = get_gt_for_image(gt_name, self.gt_path)
        image_patch = get_patch_with_index(image, patch_idx, self.patch_height_ratio, self.patch_width_ratio)
        # visualize(image, image_patch, image)
        image = image_patch
        gs, gt_gs = [], []
        if self.log_transform:
            image, gs = transform_to_log(image)
            gt, gt_gs = transform_to_log(gt)

        augmented = self.transforms(image=image, mask=gs)
        img, gs = augmented['image'], augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=gs)
            img, gs = preprocessed['image'], preprocessed['mask']
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32, device="cuda")
        gt = torch.tensor(gt, dtype=torch.float32, device='cuda')
        return img, gt, gs, gt_gs

    def __len__(self):
        return floor(len(self.image_names) / self.patch_width_ratio / self.patch_height_ratio)
