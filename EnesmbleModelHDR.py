import segmentation_models_pytorch as smp
import torch

from pytorch_metric_learning import losses
# -- MODEL --
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader

import HypNet
from Dataset import MIDataset, MIPatchedDataset
from HDRDataset import HDRDataset, HDRPatchedDataset
from Losses import BCEDiceLoss, angular_loss
from Models import get_model
from dataset_utils import visualize_tensor, transform_from_log, visualize, to_np_img
from transformation_utils import get_training_augmentation, get_preprocessing, get_validation_augmentation
import numpy as np

use_log = True
in_channels = 2 if use_log else 3
patch_size = 64

model = HypNet.HypNet(patch_height=patch_size, patch_width=patch_size, in_channels=in_channels, out_channels=in_channels)
model.cuda(0)

num_workers = 0
bs = 1


train_dataset = HDRPatchedDataset(datatype='train',
                                 transforms=get_training_augmentation(patch_size, patch_size),
                                 log_transform=use_log)  # , preprocessing=get_preprocessing(preprocessing_fn))
valid_dataset = HDRPatchedDataset(datatype='valid',
                                 transforms=get_validation_augmentation(patch_size, patch_size),
                                 log_transform=use_log)  # , preprocessing=get_preprocessing(preprocessing_fn))

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

loaders = {
    "train": train_loader,
    "valid": valid_loader
}

num_epochs = 10
log_interval = 5
logdir = "./logs/segmentation"

# model, criterion, optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)
scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
criterion1 = torch.nn.MSELoss()

# -- TRAINING --

def plot(data, gt, out_a, out_b, gs, gt_gs):
    gs = gs[0]
    gt_gs = np.array([[np.array(gt_gs[0]) for i in range(44)] for j in range(44)])
    d = to_np_img(data[0])
    d = transform_from_log(d, gs)
    gt_img = np.array([[np.array(gt[0].cpu()) for i in range(44)] for j in range(44)])
    gt_img = transform_from_log(gt_img, gt_gs)
    preda_img = np.array([[out_a[0].detach().cpu().numpy() for i in range(44)] for j in range(44)])
    preda_img = transform_from_log(preda_img, gt_gs)
    predb_img = np.array([[out_b[0].detach().cpu().numpy() for i in range(44)] for j in range(44)])
    predb_img = transform_from_log(predb_img, gt_gs)
    visualize(d, gt_img, preda_img, predb_img)

for epoch in range(num_epochs):
    for batch_idx, (data, gt, gs, gt_gs) in enumerate(train_loader):
        out_a, out_b = model(data)
        _, loss, loss_b = model.back_pass(out_a, out_b, gt, optimizer, criterion1, batch_idx < 100 and epoch == 0)
        if batch_idx % 1000 == 0:
            plot(data, gt, out_a, out_b, gs, gt_gs)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLossA: {:.6f} \t LossB: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), loss_b.item()))
        torch.cuda.empty_cache()
    for batch_idx, (data, gt, gs, gt_gs) in enumerate(valid_loader):
        if batch_idx > 1:
            break
        out_a, out_b = model(data)
        _, loss, loss_b = model.get_selection_class(out_a, out_b, gt, criterion1)
        if batch_idx % 1000 == 0:
            plot(data, gt, out_a, out_b, gs, gt_gs)
        if batch_idx % log_interval == 0:
            print('Valid Epoch: {} [{}/{} ({:.0f}%)]\tLossA: {:.6f} \t LossB: {:.6f}'.format(
                epoch, batch_idx * len(data), len(valid_loader.dataset),
                       100. * batch_idx / len(valid_loader), loss.item(), loss_b.item()))
        torch.cuda.empty_cache()

train_dataset = HDRDataset(datatype='train',
                          transforms=get_validation_augmentation(),
                          log_transform=use_log)  # , preprocessing=get_preprocessing(preprocessing_fn))
valid_dataset = HDRDataset(datatype='valid',
                          transforms=get_validation_augmentation(),
                          log_transform=use_log)  # , preprocessing=get_preprocessing(preprocessing_fn))

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

selNet = HypNet.SelUnet()
selNet.cuda(0)

num_epochs = 10
log_interval = 5
logdir = "./logs/segmentation"

# model, criterion, optimizer
optimizer = torch.optim.Adam([
    {'params': selNet.model.decoder.parameters(), 'lr': 1e-2},
    {'params': selNet.model.encoder.parameters(), 'lr': 1e-3},
])
criterion = BCEDiceLoss()

patch_width_ratio = patch_size / 640.
patch_height_ratio = patch_size / 320.

for epoch in range(num_epochs):
    for batch_idx, (data, gt) in enumerate(train_loader):
        if batch_idx > 1:
            break
        cum_loss = 0
        for img, gti in zip(data, gt):
            sel, sel_gt = selNet.select(model, img, gti, patch_height_ratio, patch_width_ratio, criterion1, is_gt_image=False)
            sel, sel_gt = sel[0], sel_gt.type(torch.FloatTensor).cuda(0)
            optimizer.zero_grad()
            loss = criterion(sel, sel_gt).mean()
            loss.backward()
            cum_loss += loss.detach()
            optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLossA: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), cum_loss.item()))
        loss = None
        torch.cuda.empty_cache()
    for batch_idx, (data, gt) in enumerate(valid_loader):
        loss = None
        for img, gti in zip(data, gt):
            sel, sel_gt = selNet.select(model, img, gti, patch_height_ratio, patch_width_ratio, criterion1, is_gt_image=False)
            sel, sel_gt = sel[0], sel_gt.type(torch.FloatTensor).cuda(0)
            optimizer.zero_grad()
            if loss is None:
                loss = criterion(sel, sel_gt).mean().detach()
            else:
                loss += criterion(sel, sel_gt).mean().detach()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLossA: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        loss = None
        torch.cuda.empty_cache()

torch.save(model.state_dict(), './models/ensemble-model-hyp-hdr')
torch.save(selNet.model.state_dict(), './models/ensemble-model-sel-hdr')
