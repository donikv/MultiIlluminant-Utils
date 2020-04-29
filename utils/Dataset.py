import os
from math import floor, ceil

import albumentations as albu
import cv2
import torch
from albumentations import pytorch as AT
from torch.utils.data import Dataset
import numpy as np

from dataset_utils import load_img_and_gt_crf_dataset, visualize, get_mask_from_gt, mask_to_image, get_patch_with_index
from transformation_utils import transform_to_log


class MIDataset(Dataset):
    def __init__(self, path: str = './data', folder: str = 'dataset_crf/lab', special_folder: str = '',
                 df: object = None, datatype: object = 'train', dataset='crf',
                 img_ids: object = None,
                 transforms: object = albu.Compose([albu.HorizontalFlip(), AT.ToTensor()]),
                 preprocessing: object = None,
                 use_mask: bool = False, use_corrected: bool = False, log_transform=False) -> object:
        self.df = df
        self.datatype = datatype
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.use_mask = use_mask
        self.path = path
        self.folder = folder
        self.log_transform = log_transform
        self.use_corrected = use_corrected
        self.dataset = dataset

        if self.use_corrected:
            self.image_names = os.listdir(f"{path}/{folder}/img_corrected_1/{special_folder}")
        else:
            if self.dataset != 'crf':
                self.image_names = os.listdir(f"{path}/{folder}/images/{special_folder}")
            else:
                self.image_names = os.listdir(f"{path}/{folder}/srgb8bit/{special_folder}")

        self.path = path
        self.folder = folder

    def __getitem__(self, idx):
        if self.dataset == 'test':
            return self.get_test_item(idx)
        else:
            return self.get_train_item(idx)

    def get_test_item(self, idx):
        image_name = self.image_names[idx]
        image = load_img_and_gt_crf_dataset(image_name, self.path, self.folder, dataset=self.dataset)
        gs = []
        if self.log_transform:
            image, gs = transform_to_log(image)
            aug = self.transforms(image=image, mask=gs)
            img, gs = aug['image'], aug['mask'].squeeze()
        else:
            aug = self.transforms(image=image, mask=image)
            img = aug['image']
        if self.preprocessing:
            if self.log_transform:
                aug = self.preprocessing(image=img, mask=gs)
                img, gs = aug['image'], aug['mask'].squeeze()
            else:
                aug = self.preprocessing(image=img, mask=image)
                img = aug['image']
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32, device="cuda")
        return img, gs

    def get_train_item(self, idx):
        image_name = self.image_names[idx]
        image, gt, mask = load_img_and_gt_crf_dataset(image_name, self.path, self.folder, use_mask=self.use_mask,
                                                      use_corrected=self.use_corrected, dataset=self.dataset)
        gs, gt_gs = [], []
        if self.log_transform:
            image, gs = transform_to_log(image)
            gs = np.expand_dims(gs, axis=2)
            gt, gt_gs = transform_to_log(gt)
            inputs = {'image': image, 'mask': mask, 'image2': gs, 'gt_gs': gt_gs, 'gt': gt}
            augmented = self.transforms(**inputs)
            gs = augmented['image2']
            gs = gs.squeeze()
            gt_gs = augmented['gt_gs']
        else:
            inputs = {'image': image, 'mask': mask, 'image2': gt}
            augmented = self.transforms(**inputs)
        img = augmented['image']
        mask = augmented['mask']
        gt = augmented['image2']
        if self.preprocessing:
            if self.log_transform:
                preprocessed = self.preprocessing(image=img, mask=mask, gt=gt, image2=gs, gt_gs=gt_gs)
                gs = preprocessed['image2']
                gt_gs = preprocessed['gt_gs']
            else:
                preprocessed = self.preprocessing(image=img, mask=mask, image2=gt)
            img = preprocessed['image']
            mask = preprocessed['mask']
            gt = preprocessed['image2']

        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32, device="cuda")
        gt = torch.tensor(gt.transpose(2, 0, 1), dtype=torch.float32, device="cuda")
        mask = np.array([[(np.array([1]) if pixel[0] > 128 else np.array([0])) for pixel in row] for row in mask])
        mask = torch.tensor(mask.transpose(2, 0, 1), dtype=torch.float32, device="cuda")
        return (img, gs), \
               mask, \
               (gt, gt_gs)

    def __len__(self):
        return len(self.image_names)


class MIPatchedDataset(MIDataset):
    def __init__(self, path: str = './data', folder: str = 'dataset_crf/lab', special_folder: str = '',
                 df: object = None, datatype: object = 'train', dataset='crf',
                 img_ids: object = None,
                 transforms: object = albu.Compose([albu.HorizontalFlip(), AT.ToTensor()]),
                 preprocessing: object = None,
                 use_mask: bool = False, use_corrected: bool = False,
                 patch_width_ratio: float = 1. / 10, patch_height_ratio: float = 1. / 10,
                 log_transform=False) -> object:
        super(MIPatchedDataset, self).__init__(path, folder, special_folder, df, datatype, dataset, img_ids, transforms,
                                               preprocessing, use_mask, use_corrected, log_transform)
        self.patch_width_ratio = patch_width_ratio
        self.patch_height_ratio = patch_height_ratio

    def __getitem__(self, idx):
        patches_per_img = ceil(1 / (self.patch_height_ratio * self.patch_width_ratio))
        image_idx = int(idx / patches_per_img)
        patch_idx = idx % patches_per_img
        image_name = self.image_names[image_idx]
        image, gt, mask = load_img_and_gt_crf_dataset(image_name, self.path, self.folder, use_mask=self.use_mask,
                                                      use_corrected=self.use_corrected, dataset=self.dataset)

        image = get_patch_with_index(image, patch_idx, self.patch_height_ratio, self.patch_width_ratio)
        mask = get_patch_with_index(mask, patch_idx, self.patch_height_ratio, self.patch_width_ratio)
        gt = get_patch_with_index(gt, patch_idx, self.patch_height_ratio, self.patch_width_ratio)

        augmented = self.transforms(image=image, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']

        gs, gt_gs = [], []
        if self.log_transform:
            img, gs = transform_to_log(img)
            gt, gt_gs = transform_to_log(gt)
            gt, gt_gs = gt.mean(0).mean(0), gt_gs.mean(0).mean(0)

        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32, device="cuda")

        gt = torch.tensor(gt, dtype=torch.float32, device="cuda")
        mask = np.array([[(np.array([1]) if pixel[0] > 128 else np.array([0])) for pixel in row] for row in mask])
        mask = torch.tensor(mask.transpose(2, 0, 1), dtype=torch.float32, device="cuda")
        return (img, gs), \
               mask, \
               (gt, gt_gs)

    def __len__(self):
        return floor(len(self.image_names) / self.patch_width_ratio / self.patch_height_ratio)
