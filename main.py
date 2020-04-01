import torch

import segmentation_models_pytorch as smp

import HypNet
from Dataset import MIDataset, MIPatchedDataset
from torch.utils.data.dataloader import DataLoader

from Models import get_model
from dataset_utils import visualize, calculate_histogram, plot_histograms, cluster, visualize_tensor
from transformation_utils import color_correct, color_correct_tensor, get_training_augmentation, \
    get_validation_augmentation, color_correct_with_mask, to_tensor


def test_model(path, images_path):
    model, _ = get_model(num_classes=2)
    model.eval()
    dataset = MIDataset(datatype='test', folder='dataset_crf/realworld', special_folder=images_path, transforms=get_validation_augmentation(), use_mask=True)
    #dataset = MIDataset(datatype='test', folder='dataset_crf/valid', transforms=get_validation_augmentation(), use_mask=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    model.load_state_dict(torch.load(path))

    for batch_idx, (data, mask, gt) in enumerate(loader):
        p_mask, label = model(data)
        visualize_tensor(data, mask, p_mask)
        _, _, center = cluster(gt.cpu())
        cimg = color_correct_with_mask(data, p_mask, center[0], center[1])
        visualize_tensor(data.cpu(), gt, p_mask, cimg)
        input("Press Enter to continue...")


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



#test_model('./models/unet-efficientnet-b0-gt', 'special')
test_hyp_sel_hdr(['./models/ensemble-model-hyp', './models/ensemble-model-sel'], '', use_log=False)
#exit(0)
#img, mask, gt = load_img_and_gt('bmug_b_r.png')
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
