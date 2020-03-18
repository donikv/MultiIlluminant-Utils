import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting


def load_img_and_gt(x, path='./data', folder='dataset_crf/lab', use_mask=True):
    """
    Return image based on image name and folder.
    """
    images_data_folder = f"{path}/{folder}/srgb8bit"
    gt_data_folder = f"{path}/{folder}/groundtruth"
    mask_data_folder = f"{path}/{folder}/gt_mask"
    if use_mask:
        mask_data_folder = f"{path}/{folder}/masks"
    image_path = os.path.join(images_data_folder, x)
    gt_path = os.path.join(gt_data_folder, x)
    mask_path = os.path.join(mask_data_folder, x)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gt = cv2.imread(gt_path)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    return img, gt, mask


def to_np_img(t: torch.tensor):
    if len(t.shape) == 4:
        t = t[0]
    return t.cpu().detach().numpy().transpose(1, 2, 0)


def mask_to_image(t: np.ndarray):
    if t.shape[-1] == 3:
        return t
    return np.array(
        [[(np.array([255, 255, 255]) if pixel[0] < pixel[1] else np.array([0, 0, 0])) for pixel in row] for row in
         t])

def visualize_tensor(image, p_mask, mask, transformed_image=None):

    t_image = to_np_img(transformed_image) if transformed_image is not None else None
    visualize(to_np_img(image).astype(int), mask_to_image(to_np_img(p_mask)), mask_to_image(to_np_img(mask)),
              transformed_image)


def visualize(image, gt, mask, transformed_image=None):
    """
    Plot image and masks.
    If two pairs of images and masks are passes, show both.
    """

    if transformed_image is None:
        f, ax = plt.subplots(2, 2, figsize=(50, 50))

        ax[0][0].imshow(image)
        ax[0][1].imshow(mask)
        ax[1][0].imshow(gt)
    else:
        f, ax = plt.subplots(2, 2, figsize=(50, 50))
        ax[1][1].imshow(transformed_image)
        ax[1][0].imshow(gt)
        ax[0][0].imshow(image)
        ax[0][1].imshow(mask)
    plt.show()


def get_mask_from_gt(gt):
    gt = gt.cpu()
    _, _, centers = cluster(gt, draw=False)
    mask = np.array([[(np.array([1, 0]) if np.linalg.norm(pixel - centers[0]) > np.linalg.norm(
        pixel - centers[1]) else np.array([0, 1])) for pixel in row] for row in gt])
    return mask


def calculate_histogram(img: np.ndarray):
    channels = [0, 1, 2]
    hist = lambda x: cv2.calcHist([img], [x], None, [100], [0, 1])
    return list(map(hist, channels))


def cluster(img: np.ndarray, draw=True):
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)
    mask = (Z > 0) & (Z < 255)
    mask = np.array(list(map(lambda x: x.all(), mask)))
    # print(mask)
    Z = Z[mask]

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    A = Z[label.ravel() == 0]
    B = Z[label.ravel() == 1]
    # C = Z[label.ravel() == 2]
    if draw:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(A[:, 0], A[:, 1], A[:, 2])
        ax.scatter(B[:, 0], B[:, 1], B[:, 2], c='r')
        ax.scatter(center[:, 0], center[:, 1], center[:, 2], s=80, c='y', marker='s')
        # ax.scatter(C[:, 0], C[:, 1], C[:, 2], c='g')
        # Now convert back into uint8, and make original image
        plt.show()
    return ret, label, center


def plot_histograms(hists):
    f, ax = plt.subplots(len(hists), 1, figsize=(50, 50))

    for idx, hist in enumerate(hists):
        ax[idx].bar(np.linspace(0, 255, len(hist)), hist.squeeze())

    plt.show()
