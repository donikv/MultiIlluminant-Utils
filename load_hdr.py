import cv2
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import utils.dataset_utils as du
from utils.HDRDataset import HDRPatchedDataset
from utils.load_mat import get_gt_for_image
from utils.transformation_utils import color_correct_fast, transform_from_log, adjust_gamma, transform_to_log

if __name__ == '__main__':

    filename = './data/dataset_568_shi_gehler/cs/chroma/data/canon_dataset/568_dataset/png/IMG_0293.png'
    im = cv2.imread(filename, cv2.COLOR_BGR2RGB)
    im2 = adjust_gamma(im, gamma=2.2)
    im3 = adjust_gamma(im, gamma=3.5)
    im4 = adjust_gamma(im, gamma=5)
    log, g = transform_to_log(im3)
    im4 = transform_from_log(log, g).astype(int)
    du.visualize(im, im2, im3, im4)

    filename_cc = './data/dataset_hdr/HDR/cs/chroma/data/Nikon_D700/HDR_MATLAB_3x3/S0560.hdr'
    im_cc = cv2.imread(filename_cc, cv2.IMREAD_ANYDEPTH)
    im_cc2 = np.array(im_cc.astype(int), dtype=np.uint8)
    im2 = adjust_gamma(im_cc2, gamma=2.2)
    log, g = transform_to_log(im2)
    im2 = transform_from_log(log, g).astype(int)
    gt = get_gt_for_image('S0560_real_illum')
    im_corrected = color_correct_fast(im_cc, gt)
    im_corrected_2 = color_correct_fast(im2, (gt*255).astype(int))
    du.visualize(im_cc, im2, im_corrected_2, im_corrected)
    log_rg, log_bg = cv2.split(log)
    plt.scatter(log_rg, log_bg, s = 2)
    plt.xlabel('Log(R/G)')
    plt.ylabel('Log(B/G)')
    plt.title('2D Log Chromaticity')
    plt.show()

    # train_dataset = HDRPatchedDataset(datatype='train',
    #                                   transforms=get_training_augmentation(100, 100),
    #                                   log_transform=False)  # , preprocessing=get_preprocessing(preprocessing_fn))
    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    #
    # for idx, (img, gt) in enumerate(train_loader):
    #     img = img.cpu()
    #     du.visualize_tensor(img[0], img[1], img[2], img[3])
    #     print(gt)
    # exit()