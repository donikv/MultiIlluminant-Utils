import cv2
import numpy as np
from torch.utils.data import DataLoader

import dataset_utils as du
from HDRDataset import HDRPatchedDataset
from load_mat import get_gt_for_image
from transformation_utils import color_correct_fast, get_training_augmentation

if __name__ == '__main__':

    train_dataset = HDRPatchedDataset(datatype='train',
                                      transforms=get_training_augmentation(100, 100),
                                      log_transform=False)  # , preprocessing=get_preprocessing(preprocessing_fn))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    for idx, (img, gt) in enumerate(train_loader):
        img = img.cpu()
        du.visualize_tensor(img[0], img[1], img[2], img[3])
        print(gt)
    exit()

    filename = './data/dataset_hdr/HDR/cs/chroma/data/Nikon_D700/HDR_MATLAB_3x3/S0010.hdr'
    im = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
    im2 = im.astype(float)

    filename_cc = './data/dataset_hdr/HDR/cs/chroma/data/Nikon_D700/HDR_MATLAB_3x3/S0560.hdr'
    im_cc = cv2.imread(filename_cc, cv2.IMREAD_ANYDEPTH)
    gt = get_gt_for_image('S0010_real_illum')
    im_corrected = color_correct_fast(im, gt)
    du.visualize(im, im_cc, im_corrected)

    # tonemapDurand = cv2.createTonemapDurand(2.2)
    # ldrDurand = tonemapDurand.process(im)
    #
    # im2_8bit = np.clip(ldrDurand * 255, 0, 255).astype('uint8')
    #
    # new_filename = filename + ".jpg"
    # cv2.imwrite(new_filename, im2_8bit)