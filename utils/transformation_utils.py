
import numpy as np
import torch

from dataset_utils import to_np_img


def color_correct(img, canonical_ill, unknown_ill=None):
    trans_mat = np.eye(3)  # .tolist()
    trans_img = []
    print(img.shape)
    for i in range(len(img)):
        trans_img.append([])
        for k in range(len(img[0])):
            div = [1, 1, 1] if unknown_ill is None else unknown_ill[i][k]
            for j in range(3):
                # print(canonical_ill[i][k][j])
                trans_mat[j][j] = div[j] / canonical_ill[i][k][j]
            trans_mat = torch.tensor(trans_mat, dtype=torch.float)
            trans_img[i].append(trans_mat @ img[i][k])

    b = torch.Tensor(img.shape)
    torch.cat(trans_img, out=b)
    return trans_img

def color_correct_fast(img, u_ill, c_ill=1/3.):
    def correct_pixel(p, ill):
        return np.multiply(p, ill)
    return np.array([np.array([correct_pixel(p, c_ill/u_ill) for p in row]) for row in img])


def color_correct_with_mask(img, mask, c1, c2):
    mask = to_np_img(mask)
    #c1 = torch.tensor(c1)
    #c2 = torch.tensor(c2)
    gt = np.array(
        [[c1 if pixel[0] < pixel[1] else c2 for pixel in row] for row in mask])
    return color_correct_tensor(img[0], torch.tensor(gt))

def color_correct_tensor(img, canonical_ill, unknown_ill=None):
    trans_mat = np.eye(3)  # .tolist()
    img = img.cpu()
    shape = canonical_ill.shape
    deltas = torch.zeros((shape[0], shape[1], 3, 3))
    print(deltas.shape)
    print(img.shape)

    for row_idx in range(canonical_ill.shape[0]):
        for pixel_idx in range(canonical_ill[row_idx].shape[0]):
            delta = torch.diag(1/canonical_ill[row_idx][pixel_idx])
            deltas[row_idx][pixel_idx] = delta
    return (deltas @ img.transpose(2, 0).transpose(1, 0).unsqueeze(-1)).squeeze()

## Preprocessing


def transform_to_log(img: np.ndarray):
    def trans(pixel):
        g = 1 if pixel[1] == 0 else pixel[1]
        # b = 0.01 if pixel[2] == 0 else pixel[2]
        # r = 0.01 if pixel[0] == 0 else pixel[0]
        return np.array([pixel[0]/g, pixel[2]/g])
    img_log = np.array(
        [[trans(pixel) for pixel in row] for row in
         img])
    return img_log

import albumentations as albu


def get_training_augmentation(x: int = 320, y: int = 640):
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        #albu.RandomGamma(p=0.75),
        albu.GridDistortion(p=0.25),
        #albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
        albu.Resize(x, y),
        #albu.Normalize(always_apply=True),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(x: int = 320, y: int = 640):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(x, y)
    ]
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)



def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype('float32')
