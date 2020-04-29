import torch

import segmentation_models_pytorch as smp

import HypNet
from Dataset import MIDataset, MIPatchedDataset
from torch.utils.data.dataloader import DataLoader

from Models import get_model, get_custom_model
from dataset_utils import visualize, calculate_histogram, plot_histograms, cluster, visualize_tensor, \
    get_center_colors, to_np_img, mask_to_image
from transformation_utils import color_correct, color_correct_tensor, get_training_augmentation, \
    get_validation_augmentation, color_correct_with_mask, to_tensor, color_correct_fast, transform_from_log, \
    get_preprocessing, get_test_augmentation
from SegmentationModel import plot
import Losses as ls
import numpy as np
import cv2


def plot(data, gs, mask, p_mask, use_log, custom_transform=lambda x: x, use_mixture=False):
    d = to_np_img(data[0])
    if use_log:
        d = cv2.split(d)
        d = np.dstack((d[0], d[1]))
        gs = gs[0]
        d = transform_from_log(d, gs)
    if d.max() > 10:
        d = d.astype(int)
    mask = mask_to_image(to_np_img(mask[0]))
    p_mask = custom_transform(mask_to_image(to_np_img(p_mask[0]), use_mixture=use_mixture))
    visualize(d, p_mask, mask=mask)


def test_custom_model(path, images_path):
    use_corrected = True
    model = get_custom_model(num_classes=1, use_sigmoid=False)
    model.eval()
    dataset = MIDataset(datatype='test', folder='dataset_relighted', special_folder=images_path,
                        transforms=get_validation_augmentation(), use_mask=False, use_corrected=use_corrected, log_transform=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.load_state_dict(torch.load(path))

    dl = ls.DiceLoss()
    for batch_idx, (data, mask, gt) in enumerate(loader):
        data, gs = data
        p_mask = model(data)
        p_mask = p_mask.clamp(0, 1)
        loss = dl(p_mask, mask).mean().detach()
        print(loss)
        plot(data, gs, mask, p_mask, True)
    torch.cuda.empty_cache()


def test_model(path, images_path, type):
    use_corrected = True
    crf = True
    model, preproc = get_model(num_classes=1, type=type)
    model.eval()
    if crf:
        dataset = MIDataset(datatype='test', folder='dataset_crf/realworld', special_folder=images_path,
                            transforms=get_test_augmentation()#, preprocessing=get_preprocessing(preproc)
                            , use_mask=False, use_corrected=use_corrected, dataset='crf')
    else:
        dataset = MIDataset(datatype='test', folder='dataset_relighted/complex2/valid', special_folder=images_path,
                        transforms=get_validation_augmentation()#, preprocessing=get_preprocessing(preproc)
                        , use_mask=False, use_corrected=use_corrected, dataset='cube')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.load_state_dict(torch.load(path))
    dl = ls.DiceLoss()

    def dice(x, y):
        return 1 - dl(x, y)

    def sigmoid(x):
        return 1 / (1 + torch.exp(x))

    for batch_idx, (data, mask, gt) in enumerate(loader):
        data, gs = data
        gt, gt_gs = gt
        p_mask, label = model(data)
        p_mask = torch.clamp(p_mask, 0, 1)
        print(dice(mask, p_mask))
        # center = get_center_colors(gt.cpu(), mask.cpu())
        # if use_corrected:
        #     gt = torch.tensor(to_tensor(color_correct_fast(to_np_img(gt), center[0])))
        #     center = get_center_colors(gt, mask.cpu())
        #
        # def filter(img):
        #     return cv2.GaussianBlur(img, (101, 101), 0)
        #
        # cimg, gt_mask = color_correct_with_mask(data, p_mask, center[0], center[1], filter)
        # if use_corrected:
        #     cimg = cimg.type(torch.IntTensor)
        # else:
        #     gt_mask = gt_mask.type(torch.IntTensor)
        # plot(data, gs, mask, p_mask, False)
        # visualize_tensor(data.cpu().type(torch.IntTensor), gt.cpu(), gt_mask, cimg)
        #
        # cimg, gt_mask = color_correct_with_mask(data, p_mask, center[1], center[0], filter)
        # if use_corrected:
        #     cimg = cimg.type(torch.IntTensor)
        # else:
        #     gt_mask = gt_mask.type(torch.IntTensor)
        # visualize_tensor(data.cpu().type(torch.IntTensor), gt.cpu(), gt_mask, cimg)
        plot(data, gs, mask, p_mask, False, use_mixture=True)
        # input("Press Enter to continue...")


def test_hyp_sel(paths, images_path, use_log=False):
    in_channels = 2 if use_log else 3
    dataset = MIDataset(datatype='test', folder='dataset_crf/realworld', special_folder=images_path,
                        transforms=get_validation_augmentation(), use_mask=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    model = HypNet.HypNet(patch_height=44, patch_width=44, in_channels=in_channels, out_channels=in_channels)
    model.cuda(0)
    model.load_state_dict(torch.load(paths[0]))

    selNet = HypNet.SelUnet()
    selNet.cuda(0)
    selNet.model.load_state_dict(torch.load(paths[1]))

    patch_width_ratio = 44. / 640
    patch_height_ratio = 44. / 320

    for batch_idx, (data, mask, gt) in enumerate(loader):
        gt = gt / 255
        for img, gti in zip(data, gt):
            final = selNet.test(model, img, gti, patch_height_ratio, patch_width_ratio)
            visualize_tensor(img.cpu(), gti, final.cpu().type(torch.IntTensor))
        torch.cuda.empty_cache()


def test_hyp_sel_hdr(paths, images_path, use_log=False):
    in_channels = 2 if use_log else 3
    dataset = MIDataset(datatype='test', folder='dataset_crf/realworld', special_folder=images_path,
                        transforms=get_validation_augmentation(), use_mask=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    model = HypNet.HypNet(patch_height=44, patch_width=44, in_channels=in_channels, out_channels=in_channels)
    model.cuda(0)
    model.load_state_dict(torch.load(paths[0]))

    selNet = HypNet.SelUnet()
    selNet.cuda(0)
    selNet.model.load_state_dict(torch.load(paths[1]))

    patch_width_ratio = 100. / 640
    patch_height_ratio = 100. / 320

    for batch_idx, (data, mask, gt) in enumerate(loader):
        gt = gt / 255.
        data = data / 255.
        for img, gti in zip(data, gt):
            gti_i = gti.mean(1).mean(1)
            final = selNet.test(model, img, gti_i, patch_height_ratio, patch_width_ratio)
            visualize_tensor(img.cpu(), gti, final.cpu())
        torch.cuda.empty_cache()


test_model('./models/unet-efficientnet-b0-gt-best-valid-cube3-26_4-x', '', type='unet')
exit(0)
test_custom_model('./models/unet-efficientnet-b0-gt-best-valid-cube3-custom2-100', '')
# test_hyp_sel_hdr(['./models/ensemble-model-hyp', './models/ensemble-model-sel'], '', use_log=False)
# exit(0)
# img, mask, gt = load_img_and_gt('bmug_b_r.png')
# print(img-gt)
# print(gt.shape)
dataset = MIPatchedDataset(datatype='train', transforms=get_training_augmentation(), use_mask=True)
img, mask, gt = dataset[10]
# hists = calculate_histogram(gt.numpy())
_, _, center = cluster(gt.cpu())
# print(center * 255)
# print(gt.shape)
# print(img.shape)
cimg = color_correct_tensor(img, gt)
# cluster(cimg.numpy())

visualize_tensor(img.cpu(), gt, mask, cimg)
