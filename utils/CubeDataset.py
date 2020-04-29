import os
import random
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


class CubeDataset(Dataset):
    def __init__(self, images_path: str = 'C:\\Users\\Donik\\Dropbox\\Donik\\fax\\10_semestar\\Diplomski\\CubeDataset\\data\\Cube+\\',
                 gt_path: str = 'C:\\Users\\Donik\\Dropbox\\Donik\\fax\\10_semestar\\Diplomski\\CubeDataset\\data\\Cube+\\cube+_gt.txt',
                 df: object = None, datatype: object = 'train',
                 transforms: object = albu.Compose([albu.HorizontalFlip(), AT.ToTensor()]),
                 preprocessing: object = None,
                 log_transform=False, illumination_known=False) -> object:
        self.df = df
        self.datatype = datatype
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.images_path = images_path
        self.gt_path = gt_path
        self.log_transform = log_transform
        self.illumination_known = illumination_known
        if datatype == 'valid':
            self.image_names = list(range(1650, 1708))
        else:
            self.image_names = list(range(1, 1650))
        self.gts = np.loadtxt(gt_path)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = self.__load(image_name, self.images_path)
        image = self.__process_image(image)
        gt = self.gts[image_name-1]
        gt_rand = self.gts[(image_name + random.randint(10, 500)) % len(self.image_names)]

        gs, gt_gs = [], []
        if self.log_transform:
            image, gs = transform_to_log(image)
            gt, gt_gs = transform_to_log(gt)
            gt_rand, _ = transform_to_log(gt_rand)

        augmented = self.transforms(image=image, mask=gs)
        img, gs = augmented['image'], augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=gs)
            img, gs = preprocessed['image'], preprocessed['mask']
        if self.illumination_known:
            shape = img.shape
            gt_img = np.array([[np.array(gt) for i in range(shape[1])] for j in range(shape[0])])
            gt_rand_img = np.array([[np.array(gt_rand) for i in range(shape[1])] for j in range(shape[0])])
            img = np.dstack((img, gt_img, gt_rand_img))
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32, device="cuda")
        gt = torch.tensor(gt, dtype=torch.float32, device='cuda')
        return img, gt, gs, gt_gs

    def __len__(self):
        return len(self.image_names)

    @staticmethod
    def __load(index, path, folder_step=200, mask_cube=True):
        start = int((index - 1) / folder_step) * folder_step + 1
        end = min(int((index - 1) / folder_step) * folder_step + folder_step, 1707)
        folder = f'PNG_{start}_{end}'
        rgb = CubeDataset.__load_png(f"{index}.PNG", directory=folder, mask_cube=mask_cube, path=path)
        return rgb

    @staticmethod
    def __load_png(name, path='./data/Cube+', directory='PNG_1_200', mask_cube=True):
        image = f"{path}/{directory}"
        image_path = os.path.join(image, name)

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        r, g, b = cv2.split(img)
        rgb = np.dstack((b, g, r))

        if mask_cube:
            for i in range(2000, rgb.shape[0]):
                for j in range(4000, rgb.shape[1]):
                    rgb[i][j] = np.zeros(3)

        return rgb

    @staticmethod
    def __process_image(img: np.ndarray, depth=14, resize=True):
        if resize:
            height, width, _ = img.shape
            img = cv2.resize(img, (int(width / 5), int(height / 5)))
        blacklevel = 2048
        saturationLevel = img.max() - 2
        img = img.astype(int)
        img = np.clip(img - blacklevel, a_min=0, a_max=np.infty).astype(int)
        m = np.where(img >= saturationLevel - blacklevel, 1, 0).sum(axis=2, keepdims=True)
        max_val = np.iinfo(np.int32).max
        m = np.where(m > 0, [0, 0, 0], [max_val, max_val, max_val])
        result = cv2.bitwise_and(img, m)

        return (result / 2 ** depth).astype(np.float)


class CubePatchedDataset(CubeDataset):
    def __init__(self, images_path: str = 'C:\\Users\\Donik\\Dropbox\\Donik\\fax\\10_semestar\\Diplomski\\CubeDataset\\data\\Cube+\\',
                 gt_path: str = 'C:\\Users\\Donik\\Dropbox\\Donik\\fax\\10_semestar\\Diplomski\\CubeDataset\\data\\Cube+\\cube+_gt.txt', df: object = None,
                 datatype: object = 'train', transforms: object = albu.Compose([albu.HorizontalFlip(), AT.ToTensor()]),
                 preprocessing: object = None, log_transform=False, illumination_known=False,
                 patch_width_ratio: float = 1. / 10, patch_height_ratio: float = 1. / 10) -> object:
        super().__init__(images_path, gt_path, df, datatype, transforms, preprocessing, log_transform, illumination_known=illumination_known)
        self.patch_width_ratio = patch_width_ratio
        self.patch_height_ratio = patch_height_ratio

    def __getitem__(self, idx):
        patches_per_img = ceil(1 / (self.patch_height_ratio * self.patch_width_ratio))
        image_idx = int(idx / patches_per_img)
        patch_idx = idx % patches_per_img

        image_name = self.image_names[idx]

        image = self.__load(image_name, self.images_path).astype(int)
        image = self.__process_image(image, resize=False)
        gt = self.gts[image_name-1]
        gt_rand = self.gts[(image_name + random.randint(10, 500)) % len(self.image_names)]
        image_patch = get_patch_with_index(image, patch_idx, self.patch_height_ratio, self.patch_width_ratio)

        image = image_patch
        gs, gt_gs = [], []
        if self.log_transform:
            image, gs = transform_to_log(image)
            gt, gt_gs = transform_to_log(gt)
            gt_rand, _ = transform_to_log(gt_rand)

        augmented = self.transforms(image=image, mask=gs)
        img, gs = augmented['image'], augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=gs)
            img, gs = preprocessed['image'], preprocessed['mask']

        if self.illumination_known:
            shape = img.shape
            gt_img = np.array([[np.array(gt) for i in range(shape[1])] for j in range(shape[0])])
            gt_rand_img = np.array([[np.array(gt_rand) for i in range(shape[1])] for j in range(shape[0])])
            if random.randint(0, 1) == 0:
                img = np.dstack((img, gt_img, gt_rand_img))
            else:
                img = np.dstack((img, gt_rand_img, gt_img))
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32, device="cuda")
        gt = torch.tensor(gt, dtype=torch.float32, device='cuda')
        return img, gt, gs, gt_gs

    def __len__(self):
        return floor(len(self.image_names) / self.patch_width_ratio / self.patch_height_ratio)
