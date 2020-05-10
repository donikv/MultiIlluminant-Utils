import torch

import segmentation_models_pytorch as smp

import HypNet
from Dataset import MIDataset, MIPatchedDataset
from torch.utils.data.dataloader import DataLoader

from Models import get_model, get_custom_model
from dataset_utils import visualize, calculate_histogram, plot_histograms, cluster, visualize_tensor, \
    get_center_colors, to_np_img, mask_to_image, load_img_and_gt_crf_dataset
from transformation_utils import color_correct, color_correct_tensor, get_training_augmentation, \
    get_validation_augmentation, color_correct_with_mask, to_tensor, color_correct_fast, transform_from_log, \
    get_preprocessing, get_test_augmentation
from SegmentationModel import plot
import Losses as ls
import numpy as np
import cv2
import utils.statistics_utils as stats


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


def test_custom_model(path, images_path, dataset):
    use_corrected = True
    model = get_custom_model(num_classes=1, use_sigmoid=False)
    model.eval()
    # dataset = MIDataset(datatype='test', folder='dataset_relighted', special_folder=images_path,
    #                     transforms=get_validation_augmentation(), use_mask=False, use_corrected=use_corrected, log_transform=True)
    # loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.load_state_dict(torch.load(path))

    test(model, dataset, images_path, None, True, use_corrected, path)

    # dl = ls.DiceLoss()
    # for batch_idx, (data, mask, gt) in enumerate(loader):
    #     data, gs = data
    #     p_mask = model(data)
    #     p_mask = p_mask.clamp(0, 1)
    #     loss = dl(p_mask, mask).mean().detach()
    #     print(loss)
    #     plot(data, gs, mask, p_mask, True)
    # torch.cuda.empty_cache()


def test_model(path, images_path, type, dataset):
    use_corrected = False
    use_log = path.endswith('log')
    num_channels = 2 if use_log else 3
    model, preproc = get_model(num_classes=1, type=type, in_channels=num_channels)
    model.eval()
    model.load_state_dict(torch.load(path))
    preproc = None if not path.endswith('preproc') else get_preprocessing(preproc)
    test(model, dataset, images_path, preproc, use_log, use_corrected, path)


def test(model, dataset, images_path, preproc, use_log, use_corrected, path):
    datatype = dataset
    folder = None
    aug = get_validation_augmentation() if use_log else get_test_augmentation()
    if dataset == 'crf':
        folder = 'dataset_crf/realworld'
        dataset = MIDataset(datatype='test', folder=folder, special_folder=images_path,
                            transforms=aug, preprocessing=preproc
                            , use_mask=False, use_corrected=use_corrected, dataset='crf', log_transform=use_log)

    elif dataset == 'test':
        folder = 'test/whatsapp'
        dataset = MIDataset(datatype='test', folder=folder, special_folder=images_path,
                            transforms=aug, preprocessing=preproc
                            , use_mask=False, use_corrected=use_corrected, dataset='test', log_transform=use_log)

    else:
        dataset = MIDataset(datatype='test', folder='dataset_relighted/valid', special_folder=images_path,
                        transforms=get_validation_augmentation(), preprocessing=preproc
                        , use_mask=False, use_corrected=use_corrected, dataset='cube', log_transform=use_log)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    dl = ls.DiceLoss()

    def dice(x, y):
        return 1 - dl(x, y)

    def sigmoid(x):
        return 1 / (1 + torch.exp(-x))

    if datatype == 'test':
        for idx,  (name, data, gs) in enumerate(loader):
            p_mask, label = model(data)
            sig_mask = sigmoid(p_mask)
            p_mask = torch.clamp(p_mask, 0, 1)
            name = name[0]
            image = load_img_and_gt_crf_dataset(name, folder=folder, dataset=datatype)
            fx = image.shape[1] / p_mask.shape[3]
            fy = image.shape[0] / p_mask.shape[2]
            rot_mask = cv2.resize(mask_to_image(to_np_img(sig_mask)), (0, 0), fx=fx, fy=fy)
            if rot_mask.shape[0] > rot_mask.shape[1]:
                rot_mask = rot_mask.transpose((1, 0, 2))
            cv2.imwrite(f'data/{folder}/masks/{name}', rot_mask)
            plot(data, gs, sig_mask, p_mask, use_log, use_mixture=True)
            torch.cuda.empty_cache()
        return
    dices = []
    sig_dices = []
    for batch_idx, (data, mask, gt) in enumerate(loader):
        data, gs = data
        gt, gt_gs = gt
        p_mask, label = model(data)
        p_mask_clamp = torch.clamp(p_mask, 0, 1)
        sig_mask = sigmoid(p_mask)
        plot(data, gs, mask, sig_mask, use_log, use_mixture=True)
        dc = dice(mask, p_mask_clamp) if use_corrected else max(dice(mask, p_mask_clamp), dice(1-mask, p_mask_clamp))
        dices.append(dc.item())
        dc_sig = dice(mask, sig_mask) if use_corrected else max(dice(mask, sig_mask), dice(1-mask, sig_mask))
        sig_dices.append(dc_sig.item())
        # print(dc)
    print(folder, path, use_corrected)
    print(f'Mean: {np.array(dices).mean()}\t Trimean: {stats.trimean(dices)}\t Median: {stats.median(dices)}')
    print(f'Mean: {np.array(sig_dices).mean()}\t Trimean: {stats.trimean(sig_dices)}\t Median: {stats.median(sig_dices)}')
    print('--------------------------------------------------------------------------------')



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


type = 'unet'
dataset = 'test'
# test_model('models/unet-efficientnet-b0-gt-best-valid-cube6-06_5-log', '', type=type, dataset=dataset)
# print('Testing model 2')
# test_model('models/unet-efficientnet-b0-gt-best-valid-cube3-26_4-x', '', type=type, dataset=dataset)
# print('Testing model 3')
# test_model('models/unet-efficientnet-b0-gt-best-valid-cube-comb-04_5', '', type=type, dataset=dataset)
test_custom_model('./models/unet-efficientnet-b0-gt-best-valid-cube3-custom2-100', '', dataset=dataset)
exit(0)
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
