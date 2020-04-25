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


def color_correct_fast(img, u_ill, c_ill=1 / 3.):
    def correct_pixel(p, ill):
        return np.clip(np.multiply(p, ill), a_min=0, a_max=255)

    return np.array([np.array([correct_pixel(p, c_ill / u_ill) for p in row]) for row in img])


def color_correct_with_mask(img, mask, c1, c2, filter=lambda x: x):
    mask = to_np_img(mask)
    mask = mask / (np.max(mask))
    mask = np.clip(mask, 0, 1)
    # c1 = torch.tensor(c1)
    # c2 = torch.tensor(c2)
    # gt = np.array(
    #     [[c2 * pixel[0] + c1 * (1 - pixel[0]) for pixel in row] for row in mask])
    gt = np.array(
        [[c1 * (1 - pixel[0]) + c2 * pixel[0] for pixel in row] for row in mask])
    gt = filter(gt)
    gt = torch.tensor(gt)
    return color_correct_tensor(img[0], gt), gt


def color_correct_tensor(img, canonical_ill, unknown_ill=None):
    trans_mat = np.eye(3)  # .tolist()
    img = img.cpu()
    shape = canonical_ill.shape
    deltas = torch.zeros((shape[0], shape[1], 3, 3))
    print(deltas.shape)
    print(img.shape)

    for row_idx in range(canonical_ill.shape[0]):
        for pixel_idx in range(canonical_ill[row_idx].shape[0]):
            delta = torch.diag(1 / canonical_ill[row_idx][pixel_idx] / 3)
            deltas[row_idx][pixel_idx] = delta
    return (deltas @ img.transpose(2, 0).transpose(1, 0).unsqueeze(-1)).squeeze()


# GAMMA CORRECTION
import cv2


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


## Preprocessing


def transform_to_log(img: np.ndarray):
    if len(img.shape) != 1:
        r, g, b = cv2.split(img)
    else:
        r, g, b = img[0], img[1], img[2]
    # "Normalized" channels
    # NOTE: np.ma is the masked array library. It automatically masks
    #       inf and nan answers from result
    n_r = np.ma.divide(1.*r, g)
    n_b = np.ma.divide(1.*b, g)

    log_rg = np.ma.log( n_r )
    log_bg = np.ma.log( n_b )
    if len(img.shape) != 1:
        return np.dstack((np.ma.filled(log_rg, 0), np.ma.filled(log_bg, 0))), g
    else:
        return np.array([log_rg, log_bg]), g

def transform_from_log(log, g):
    log_rg, log_bg = cv2.split(log)
    n_r_2 = np.ma.exp(log_rg)
    n_b_2 = np.ma.exp(log_bg)
    r2 = np.ma.multiply(n_r_2, g)
    b2 = np.ma.multiply(n_b_2, g)
    r2, b2 = np.ma.filled(r2, 0), np.ma.filled(b2, 0)
    img = np.dstack((r2, g, b2))
    return img

import albumentations as albu


def get_training_augmentation(x: int = 320, y: int = 640):
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        # albu.RandomGamma(p=0.75),
        albu.GridDistortion(p=0.25),
        albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
        albu.Resize(x, y),
        # albu.Normalize(always_apply=True),
    ]
    return albu.Compose(train_transform, additional_targets={"image2": "image"})


def get_validation_augmentation(x: int = 320, y: int = 640):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(x, y)
    ]
    return albu.Compose(test_transform, additional_targets={"image2": "image"})


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
