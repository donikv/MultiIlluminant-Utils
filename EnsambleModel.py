import segmentation_models_pytorch as smp
import torch

# -- MODEL --
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader

import HypNet
from Dataset import MIDataset, MIPatchedDataset
from Losses import BCEDiceLoss, angular_loss
from Models import get_model
from dataset_utils import visualize_tensor
from transformation_utils import get_training_augmentation, get_preprocessing, get_validation_augmentation

use_log = False
in_channels = 2 if use_log else 3

model = HypNet.HypNet(patch_height=44, patch_width=44, in_channels=in_channels, out_channels=in_channels)
model.cuda(0)

num_workers = 0
bs = 16

train_dataset = MIPatchedDataset(datatype='train',
                          transforms=get_training_augmentation(44, 44), use_mask=True, log_transform=use_log)  # , preprocessing=get_preprocessing(preprocessing_fn))
valid_dataset = MIPatchedDataset(folder="dataset_crf/valid", datatype='valid',
                          transforms=get_validation_augmentation(44, 44), use_mask=True, log_transform=use_log)  # , preprocessing=get_preprocessing(preprocessing_fn))

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

loaders = {
    "train": train_loader,
    "valid": valid_loader
}

num_epochs = 5
log_interval = 5
logdir = "./logs/segmentation"

# model, criterion, optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-3)
scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
criterion1 = smp.utils.losses.MSELoss()

# -- TRAINING --

for epoch in range(num_epochs):
    for batch_idx, (data, mask, gt) in enumerate(train_loader):
        out_a, out_b = model(data)
        gt = gt / 255
        gt = gt.mean(2).mean(2)
        _, loss, loss_b = model.back_pass(out_a, out_b, gt, optimizer, criterion1, batch_idx < 500 and epoch == 0)
        if batch_idx == 0:
            gt_img = torch.stack([torch.stack([gt[0] for i in range(44)]) for j in range(44)])
            preda_img = torch.stack([torch.stack([out_a[0] for i in range(44)]) for j in range(44)])
            predb_img = torch.stack([torch.stack([out_b[0] for i in range(44)]) for j in range(44)])
            visualize_tensor(data, gt_img.cpu(), preda_img.cpu().detach(), predb_img.cpu().detach())
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLossA: {:.6f} \t LossB: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), loss_b.item()))
        torch.cuda.empty_cache()
    for batch_idx, (data, mask, gt) in enumerate(valid_loader):
        out_a, out_b = model(data)
        gt = gt / 255
        gt = gt.mean(2).mean(2)
        _, loss, loss_b = model.get_selection_class(out_a, out_b, gt, criterion1)
        if batch_idx == 0:
            gt_img = torch.stack([torch.stack([gt[0] for i in range(44)]) for j in range(44)])
            preda_img = torch.stack([torch.stack([out_a[0] for i in range(44)]) for j in range(44)])
            predb_img = torch.stack([torch.stack([out_b[0] for i in range(44)]) for j in range(44)])
            visualize_tensor(data, gt_img.cpu(), preda_img.cpu().detach(), predb_img.cpu().detach())
        if batch_idx == 0:
            print(gt-out_a, gt-out_b)
        if batch_idx % log_interval == 0:
            print('Valid Epoch: {} [{}/{} ({:.0f}%)]\tLossA: {:.6f} \t LossB: {:.6f}'.format(
                epoch, batch_idx * len(data), len(valid_loader.dataset),
                       100. * batch_idx / len(valid_loader), loss.item(), loss_b.item()))
        torch.cuda.empty_cache()


train_dataset = MIDataset(datatype='train',
                          transforms=get_training_augmentation(), use_mask=True, log_transform=use_log)  # , preprocessing=get_preprocessing(preprocessing_fn))
valid_dataset = MIDataset(folder="dataset_crf/valid", datatype='valid',
                          transforms=get_validation_augmentation(), use_mask=True, log_transform=use_log)  # , preprocessing=get_preprocessing(preprocessing_fn))

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

patch_width_ratio = 44. / 640
patch_height_ratio = 44. / 320

for epoch in range(num_epochs):
    for batch_idx, (data, mask, gt) in enumerate(train_loader):
        gt = gt / 255
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
    for batch_idx, (data, mask, gt) in enumerate(valid_loader):
        gt = gt / 255
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
        loss = None
        torch.cuda.empty_cache()

torch.save(model.state_dict(), './models/ensemble-model-hyp')
torch.save(selNet.model.state_dict(), './models/ensemble-model-sel')
