import os

import albumentations as albu
import torch
from albumentations import pytorch as AT
from torch.utils.data import Dataset
import numpy as np

from dataset_utils import load_img_and_gt, visualize


class MIDataset(Dataset):
    def __init__(self, path: str = './data', folder: str = 'dataset_crf/lab', df: object = None, datatype: object = 'train',
                 img_ids: object = None,
                 transforms: object = albu.Compose([albu.HorizontalFlip(), AT.ToTensor()]),
                 preprocessing: object = None) -> object:
        self.df = df
        self.datatype = datatype
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

        if self.datatype != 'test':
            self.gt_names = os.listdir(f"{path}/{folder}/groundtruth/")

        self.image_names = os.listdir(f"{path}/{folder}/srgb8bit/")

        self.path = path
        self.folder = folder


    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image, gt, mask = load_img_and_gt(image_name, self.path, self.folder)
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
        #mask = np.apply_along_axis(lambda x: x[], 2, mask)
        mask = np.array([[(np.array([1, 0]) if pixel[0] > 128 else np.array([0, 1])) for pixel in row] for row in mask])
        mask = torch.tensor(mask.transpose(2, 0, 1), dtype=torch.float32, device="cuda")
        gt = torch.tensor(gt, dtype=torch.float32, device="cuda")
        return img, \
               mask,\
               gt

    def __len__(self):
        return len(self.image_names)
