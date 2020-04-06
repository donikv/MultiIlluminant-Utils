import os

import cv2
import torch
from torch.utils.data.dataloader import DataLoader

import albumentations as albu
from albumentations import pytorch as AT
import numpy as np
import dataset_utils as du
import transformation_utils as tu

from Dataset import MIDataset

def create_corrected_image(img: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    result = [(i,j) if (mask[i][j] == 0).all() else None for j in range(mask.shape[1]) for i in range(mask.shape[0])]
    result = list(filter(lambda x: x is not None, result))
    gt_sum = [0,0,0]
    for ind in result:
        gt_sum[0] += gt[ind[0], ind[1]][0]
        gt_sum[2] += gt[ind[0], ind[1]][2]
        gt_sum[1] += gt[ind[0], ind[1]][1]
    gt_sum[0] = gt_sum[0] / len(result)
    gt_sum[1] = gt_sum[1] / len(result)
    gt_sum[2] = gt_sum[2] / len(result)
    center = np.array(gt_sum) / 255
    corrected = tu.color_correct_fast(img, u_ill=center)
    return corrected

if __name__ == '__main__':
    make_mask = True
    path = './data'
    folder = 'dataset_crf/realworld'
    special_folder = ''
    image_names = os.listdir(f"{path}/{folder}/srgb8bit/{special_folder}")
    for name in image_names:
        image, gt, mask = du.load_img_and_gt_crf_dataset(name, path, folder, use_mask=True)
        if make_mask:
            filename = f"{path}/{folder}/gt_mask/{name}"
            mask = du.mask_to_image(du.get_mask_from_gt(gt))
            cv2.imwrite(filename, mask)
        corrected = create_corrected_image(image, gt, mask).astype(int)
        du.visualize(image, gt, mask, corrected)
        r,g,b = cv2.split(corrected)
        corrected = np.dstack((b,g,r))
        filename = f"{path}/{folder}/img_corrected_1/{name}"
        cv2.imwrite(filename, corrected)