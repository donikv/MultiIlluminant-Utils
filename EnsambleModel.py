import segmentation_models_pytorch as smp
import torch

from pytorch_metric_learning import losses
# -- MODEL --
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader

import utils.HypNet as HypNet
from utils.Dataset import MIDataset, MIPatchedDataset
from utils.Losses import BCEDiceLoss, angular_loss
from utils.Models import get_model
from utils.dataset_utils import visualize_tensor
import numpy as np
import cv2
from utils.transformation_utils import get_training_augmentation, get_preprocessing, get_validation_augmentation

use_log = True
in_channels = 2 if use_log else 3


patch_size = 64
model = HypNet.HypNet(patch_height=patch_size, patch_width=patch_size, in_channels=in_channels, out_channels=in_channels)
model.cuda(0)

num_workers = 0
bs = 64

train_dataset = MIPatchedDataset(folder='dataset_relighted', datatype='train', dataset='cube',
                                 transforms=get_training_augmentation(patch_size, patch_size), use_mask=False,
                                 log_transform=use_log)  # , preprocessing=get_preprocessing(preprocessing_fn))
valid_dataset = MIPatchedDataset(folder="dataset_relighted/valid", datatype='valid', dataset='cube',
                                 transforms=get_validation_augmentation(patch_size, patch_size), use_mask=False,
                                 log_transform=use_log)  # , preprocessing=get_preprocessing(preprocessing_fn))

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

loaders = {
    "train": train_loader,
    "valid": valid_loader
}

num_epochs = 20
log_interval = 5
logdir = "./logs/segmentation"

# model, criterion, optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-2)
scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
criterion1 = torch.nn.MSELoss()

# -- TRAINING --

from dataset_utils import visualize_tensor, transform_from_log, visualize, to_np_img
def plot(data, gt, out_a, out_b, gs, gt_gs):
    gs = gs[0]
    gt_gs = np.array([[np.array(gt_gs[0]) for i in range(64)] for j in range(64)])
    d = to_np_img(data[0])
    d = cv2.split(d)
    d = np.dstack((d[0], d[1]))
    d = transform_from_log(d, gs).astype(int)
    gt_img = np.array([[np.array(gt[0].cpu()) for i in range(64)] for j in range(64)])
    gt_img = transform_from_log(gt_img, gt_gs).astype(int)
    preda_img = np.array([[out_a[0].detach().cpu().numpy() for i in range(64)] for j in range(64)])
    preda_img = transform_from_log(preda_img, gt_gs).astype(int)
    predb_img = np.array([[out_b[0].detach().cpu().numpy() for i in range(64)] for j in range(64)])
    predb_img = transform_from_log(predb_img, gt_gs).astype(int)
    visualize(d, gt_img, preda_img, predb_img)


min_valid_loss = 0
for epoch in range(num_epochs):
    for batch_idx, (data, mask, gt) in enumerate(train_loader):
        data, gs = data
        gt, gt_gs = gt
        out_a, out_b = model(data)
        _, loss, loss_b = model.back_pass(out_a, out_b, gt, optimizer, criterion1, batch_idx < 2400 and epoch == 0)
        if batch_idx % 1000 == 0:
            plot(data, gt, out_a, out_b, gs, gt_gs)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLossA: {:.6f} \t LossB: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), loss_b.item()))
        torch.cuda.empty_cache()

    cum_loss = 0
    for batch_idx, (data, mask, gt) in enumerate(valid_loader):
        data, gs = data
        gt, gt_gs = gt
        out_a, out_b = model(data)
        _, loss, loss_b = model.get_selection_class(out_a, out_b, gt, criterion1)
        if batch_idx % 1000 == 0:
            plot(data, gt, out_a, out_b, gs, gt_gs)
        if batch_idx % log_interval == 0:
            print('Valid Epoch: {} [{}/{} ({:.0f}%)]\tLossA: {:.6f} \t LossB: {:.6f}'.format(
                epoch, batch_idx * len(data), len(valid_loader.dataset),
                       100. * batch_idx / len(valid_loader), loss.item(), loss_b.item()))
        torch.cuda.empty_cache()
        cum_loss += min(loss.mean().detach(), loss_b.mean().detach())
    print('Valid Epoch: {} \tCumulative loss {} \tMinimum loss {}'.format(
                epoch, cum_loss, min_valid_loss))
    if epoch == 0:
        min_valid_loss = cum_loss
    if min_valid_loss > cum_loss:
        min_valid_loss = cum_loss
        torch.save(model.state_dict(), './models/ensemble-model-hyp-cube-best')


train_dataset = MIDataset(folder='dataset_relighted', datatype='train', dataset='cube',
                          transforms=get_training_augmentation(), use_mask=True,
                          log_transform=use_log)  # , preprocessing=get_preprocessing(preprocessing_fn))
valid_dataset = MIDataset(folder="dataset_relighted/valid", datatype='valid', dataset='cube',
                          transforms=get_validation_augmentation(), use_mask=True,
                          log_transform=use_log)  # , preprocessing=get_preprocessing(preprocessing_fn))

bs = 16

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

selNet = HypNet.SelUnet()
selNet.cuda(0)

num_epochs = 20
log_interval = 5
logdir = "./logs/segmentation"

# model, criterion, optimizer
optimizer = torch.optim.Adam([
    {'params': selNet.model.decoder.parameters(), 'lr': 1e-2},
    {'params': selNet.model.encoder.parameters(), 'lr': 1e-3},
])
criterion = BCEDiceLoss()

patch_width_ratio = patch_size / 640
patch_height_ratio = patch_size / 320

for epoch in range(num_epochs):
    for batch_idx, (data, mask, gt) in enumerate(train_loader):
        cum_loss = 0
        for img, gti in zip(data, gt):
            sel, sel_gt = selNet.select(model, img, gti, patch_height_ratio, patch_width_ratio, criterion1)
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
    cum_loss = 0
    for batch_idx, (data, mask, gt) in enumerate(valid_loader):
        loss = None
        for img, gti in zip(data, gt):
            sel, sel_gt = selNet.select(model, img, gti, patch_height_ratio, patch_width_ratio, criterion1)
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
        cum_loss += loss.mean().detach()
        loss = None
    print('Valid Epoch: {} \tCumulative loss {} \tMinimum loss {}'.format(
                epoch, cum_loss, min_valid_loss))
    if epoch == 0:
        min_valid_loss = cum_loss
    if min_valid_loss > cum_loss:
        min_valid_loss = cum_loss
        torch.save(model.state_dict(), './models/ensemble-model-hyp-cube-best')
        torch.cuda.empty_cache()

torch.save(selNet.model.state_dict(), './models/ensemble-model-sel-cube')
