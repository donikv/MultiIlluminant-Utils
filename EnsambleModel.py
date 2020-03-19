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


model = HypNet.HypNet(patch_height=44, patch_width=44)
model.cuda(0)

num_workers = 0
bs = 16
train_dataset = MIPatchedDataset(datatype='train',
                          transforms=get_training_augmentation(44, 44), use_mask=True)  # , preprocessing=get_preprocessing(preprocessing_fn))
valid_dataset = MIPatchedDataset(folder="dataset_crf/valid", datatype='valid',
                          transforms=get_validation_augmentation(44, 44), use_mask=True)  # , preprocessing=get_preprocessing(preprocessing_fn))

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

loaders = {
    "train": train_loader,
    "valid": valid_loader
}

num_epochs = 19
log_interval = 5
logdir = "./logs/segmentation"

# model, criterion, optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-2)
scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
criterion1 = smp.utils.losses.MSELoss()

# -- TRAINING --

for epoch in range(num_epochs):
    for batch_idx, (data, mask, gt) in enumerate(train_loader):
        out_a, out_b = model(data)
        #gt = gt / 255
        gt = gt.mean(2).mean(2)
        _, loss, loss_b = model.back_pass(out_a, out_b, gt, optimizer, criterion1)
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
        #gt = gt / 255
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
                          transforms=get_training_augmentation(32, 64), use_mask=True)  # , preprocessing=get_preprocessing(preprocessing_fn))
valid_dataset = MIDataset(folder="dataset_crf/valid", datatype='valid',
                          transforms=get_validation_augmentation(32, 64), use_mask=True)  # , preprocessing=get_preprocessing(preprocessing_fn))

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)
torch.save(model.state_dict(), './models/ensemble-model-hyp')