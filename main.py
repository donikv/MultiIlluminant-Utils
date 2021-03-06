import torch

import segmentation_models_pytorch as smp

import utils.HypNet as HypNet
from utils.Dataset import MIDataset, MIPatchedDataset
from torch.utils.data.dataloader import DataLoader
import os

from utils.Models import get_model, get_custom_model
from utils.dataset_utils import visualize, calculate_histogram, plot_histograms, cluster, visualize_tensor, \
    get_center_colors, to_np_img, mask_to_image, load_img_and_gt_crf_dataset
from utils.transformation_utils import color_correct, color_correct_tensor, get_training_augmentation, \
    get_validation_augmentation, color_correct_with_mask, to_tensor, color_correct_fast, transform_from_log, \
    get_preprocessing, get_test_augmentation
from SegmentationModel import plot
import utils.Losses as ls
import numpy as np
import cv2
import utils.statistics_utils as stats


def plot(data, gs, mask, p_mask, use_log, reg, custom_transform=lambda x: x, use_mixture=False):
    d = to_np_img(data[0])
    if use_log:
        d = cv2.split(d)
        d = np.dstack((d[0], d[1]))
        gs = gs[0]
        d = transform_from_log(d, gs)
    if d.max() > 10:
        d = d.astype(int)
    if reg:
        p_mask = to_np_img(p_mask[0])
        p_mask = p_mask.astype(np.uint8)
        p_mask = cv2.cvtColor(to_np_img(p_mask), cv2.COLOR_LUV2RGB)
        mask = to_np_img(mask[0])
        mask = mask.astype(np.uint8)
        mask = cv2.cvtColor(to_np_img(mask), cv2.COLOR_LUV2RGB)
    else:
        mask = mask_to_image(to_np_img(mask[0]))
        p_mask = custom_transform(mask_to_image(to_np_img(p_mask[0]), use_mixture=use_mixture))
    visualize(d, p_mask, mask=mask)


def test_custom_model(path, images_path, dataset):
    use_corrected = False
    model = get_custom_model(num_classes=1, use_sigmoid=False)
    model.eval()
    # dataset = MIDataset(datatype='test', folder='dataset_relighted', special_folder=images_path,
    #                     transforms=get_validation_augmentation(), use_mask=False, use_corrected=use_corrected, log_transform=True)
    # loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.load_state_dict(torch.load(path))

    test(model, dataset, images_path, None, True, use_corrected, path, True, False)

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
    test(model, dataset, images_path, preproc, use_log, use_corrected, path, False, False)


def test_reg_model(path, images_path, type, dataset):
    use_log = path.endswith('log')
    num_channels = 2 if use_log else 3
    model, preproc = get_model(num_classes=3, type=type, in_channels=num_channels)
    model.eval()
    model.load_state_dict(torch.load(path))
    preproc = None if not path.endswith('preproc') else get_preprocessing(preproc)
    test(model, dataset, images_path, preproc, use_log, False, path, False, True)


def test(model, dataset, images_path, preproc, use_log, use_corrected, path, custom, reg):
    datatype = dataset
    dt = 'valid'
    folder = None
    path = './data'
    aug = get_validation_augmentation() if use_log else get_test_augmentation()
    if dataset == 'crf':
        folder = 'dataset_crf/realworld'
        dataset = MIDataset(datatype=dt, folder=folder, special_folder=images_path,
                            transforms=aug, preprocessing=preproc
                            , use_mask=False, use_corrected=use_corrected, dataset='crf', log_transform=use_log)

    elif dataset == 'test':
        folder = 'test/whatsapp'
        dataset = MIDataset(datatype='test', folder=folder, special_folder=images_path,
                            transforms=aug, preprocessing=preproc
                            , use_mask=False, use_corrected=use_corrected, dataset='test', log_transform=use_log)

    elif dataset == 'projector_test':
        folder = 'both'
        path = 'G:\\fax\\diplomski\\Datasets\\third\\ambient'
        dataset = MIDataset(path=path, datatype='test', folder=folder, special_folder=images_path,
                            transforms=aug, preprocessing=preproc
                            , use_mask=False, use_corrected=use_corrected, dataset='test', log_transform=use_log)

    else:
        folder = 'dataset_relighted/valid'
        dataset = MIDataset(datatype=dt, folder='dataset_relighted/valid', special_folder=images_path,
                        transforms=get_validation_augmentation(), preprocessing=preproc
                        , use_mask=False, use_corrected=use_corrected, dataset='cube', log_transform=use_log)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    dl = ls.DiceLoss()

    def dice(x, y):
        return 1 - dl(x, y)

    def sigmoid(x):
        return 1 / (1 + torch.exp(-x))


    def save_mask(name, mask):
        image, _, _ = load_img_and_gt_crf_dataset(name, path=path, folder=folder, dataset=datatype, use_corrected=use_corrected, rotate=False, use_mask=False)
        if image.shape[0] > image.shape[1]:
            fx = image.shape[0] / mask.shape[3]
            fy = image.shape[1] / mask.shape[2]
        else:
            fx = image.shape[1] / mask.shape[3]
            fy = image.shape[0] / mask.shape[2]
        rot_mask = cv2.resize(mask_to_image(to_np_img(mask)), (0, 0), fx=fx, fy=fy)
        if image.shape[0] > image.shape[1]:
            rot_mask = cv2.flip(rot_mask, -1)
            rot_mask = cv2.flip(rot_mask, 0)
            rot_mask = cv2.rotate(rot_mask, cv2.ROTATE_180)
        fld = f'{path}/{folder}/pmasks6{"-cor" if use_corrected else ""}{"-custom" if custom else ""}{"-reg" if reg else ""}'
        if not os.path.exists(fld):
            os.mkdir(fld)
        rot_mask = cv2.cvtColor(rot_mask, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{fld}/{name}', rot_mask)


    if datatype == 'test' or dt == 'test':
        for idx, (name, data, gs) in enumerate(loader):
            if not custom:
                p_mask, label = model(data)
            else:
                p_mask = model(data)
            if p_mask.max() < 3:
                p_mask = torch.clamp(p_mask, 0, 1)
            sig_mask = sigmoid(p_mask)
            save_mask(name[0], p_mask)
            plot(data, gs, sig_mask, p_mask, use_log, reg, use_mixture=True)
            torch.cuda.empty_cache()
        return
    dices = []
    sig_dices = []
    for batch_idx, (data, mask, gt) in enumerate(loader):
        data, gs = data
        if not custom:
            p_mask, label = model(data)
        else:
            p_mask = model(data)
        # save_mask(str(batch_idx), sig_mask)
        if not reg:
            p_mask_clamp = torch.clamp(p_mask, 0, 1)
            sig_mask = sigmoid(p_mask)
            plot(data, gs, mask, sig_mask, use_log, reg, use_mixture=True)
            dc = dice(mask, p_mask_clamp) if use_corrected else max(dice(mask, p_mask_clamp), dice(1-mask, p_mask_clamp))
            dices.append(dc.item())
            dc_sig = dice(mask, sig_mask) if use_corrected else max(dice(mask, sig_mask), dice(1-mask, sig_mask))
            sig_dices.append(dc_sig.item())
        else:
            plot(data, gs, p_mask, gt, use_log, reg, use_mixture=True)
        # print(dc)
    if not reg:
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
# test_custom_model('./models/unet-custom-gt-best-valid-cube6-11_5-log', '', dataset=dataset)
# test_model('models/fpn-effb0-gt-best-valid-cube6-18_5-log', '', type='fpn', dataset=dataset)
# print('Testing model 2')
# test_model('models/unet-efficientnet-b0-gt-best-valid-cube3-26_4-x', '', type=type, dataset=dataset)
# print('Testing model 3')
# test_model('models/unet-efficientnet-b0-gt-best-valid-cube-comb-04_5', '', type=type, dataset=dataset)
# test_model('models/unet-effb0-gt-best-valid-cube-comb-15_5-log', '', type=type, dataset=dataset) # NAUCIO BRIGHTNESE?
# test_custom_model('./models/unet-efficientnet-b0-gt-best-valid-cube3-custom2', '', dataset=dataset)
test_reg_model('models/best/reg-unet-efficientnet-b2-gt-best-valid-cube-gradient_12500-14_6-log', '', type=type, dataset=dataset)


# test_model('models/unet-efficientnet-b0-gt-best-valid-cube', '', type=type, dataset=dataset)
# test_model('models/unet-efficientnet-b0-gt-best-valid-cube2', '', type=type, dataset=dataset)
# test_model('models/unet-efficientnet-b0-gt-best-valid-cube3-26_4-x', '', type=type, dataset=dataset)
# test_model('models/unet-efficientnet-b0-gt-best-valid-cube5-30_4-preproc', '', type=type, dataset=dataset)
# test_model('models/unet-efficientnet-b0-gt-best-valid-cube6-06_5-log', '', type=type, dataset=dataset)
# test_model('models/unet-efficientnet-b0-gt-best-valid-cube6-06_5-log', '', type=type, dataset=dataset)

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
