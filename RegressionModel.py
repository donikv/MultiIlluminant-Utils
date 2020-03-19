import segmentation_models_pytorch as smp
import torch

# -- MODEL --
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader

from Dataset import MIDataset
from Losses import BCEDiceLoss
from Models import get_model
from dataset_utils import visualize_tensor
from transformation_utils import get_training_augmentation, get_preprocessing, get_validation_augmentation


model, preprocessing_fn = get_model(num_classes=3)

num_workers = 0
bs = 4
train_dataset = MIDataset(datatype='train',
                          transforms=get_training_augmentation(), use_mask=True)  # , preprocessing=get_preprocessing(preprocessing_fn))
valid_dataset = MIDataset(folder="dataset_crf/valid", datatype='valid',
                          transforms=get_validation_augmentation(), use_mask=True)  # , preprocessing=get_preprocessing(preprocessing_fn))

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
optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': 1e-2},
    {'params': model.encoder.parameters(), 'lr': 1e-3},
])
scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
criterion1 = smp.utils.losses.L1Loss()

# -- TRAINING --

for epoch in range(num_epochs):
    for batch_idx, (data, mask, gt) in enumerate(train_loader):
        p_mask, label = model(data)
        optimizer.zero_grad()
        gt = gt / 255
        loss = criterion1(p_mask, gt).mean()
        #         loss += criterion2(label, )
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        torch.cuda.empty_cache()
    for batch_idx, (data, mask, gt) in enumerate(valid_loader):
        p_mask, label = model(data)
        gt = gt / 255
        if batch_idx == 0:
            visualize_tensor(data[0], gt[0], p_mask[0])
        optimizer.zero_grad()
        loss = criterion1(p_mask, gt).mean()
        #         loss += criterion2(label, )
        if batch_idx % log_interval == 0:
            print('Valid Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(valid_loader.dataset),
                       100. * batch_idx / len(valid_loader), loss.item()))
        torch.cuda.empty_cache()
torch.save(model.state_dict(), './models/unet-efficientnet-b0-reg-l1')
