import segmentation_models_pytorch as smp
import torch

# -- MODEL --
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader

from Dataset import MIDataset
from Losses import BCEDiceLoss
from dataset_utils import visualize_tensor
from transformation_utils import get_training_augmentation, get_preprocessing, get_validation_augmentation

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'

ACTIVATION = None
aux_params=dict(
    pooling='max',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    activation='sigmoid',      # activation function, default is None
    classes=2,                 # define number of output labels
)
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=2,
    activation=ACTIVATION,
    aux_params=aux_params
)
model.cuda(0)
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

num_workers = 0
bs = 1
train_dataset = MIDataset(datatype='train', transforms=get_training_augmentation())#, preprocessing=get_preprocessing(preprocessing_fn))
valid_dataset = MIDataset(folder="dataset_crf/valid", datatype='valid', transforms=get_validation_augmentation()) #, preprocessing=get_preprocessing(preprocessing_fn))

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
criterion1 = smp.utils.losses.DiceLoss(eps=1.)
criterion2 = BCEDiceLoss()

# -- TRAINING --

for epoch in range(num_epochs):
    for batch_idx, (data, mask, gt) in enumerate(train_loader):
        p_mask, label = model(data)
        optimizer.zero_grad()
        loss = criterion2(p_mask, mask).mean()
        if epoch == 10 and batch_idx == 0:
            visualize_tensor(data, mask, p_mask)
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

        if epoch == 10:
            visualize_tensor(data, mask, p_mask)
            exit()
        optimizer.zero_grad()
        loss = criterion2(p_mask, mask).mean()
#         loss += criterion2(label, )
        if batch_idx % log_interval == 0:
            print('Valid Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(valid_loader.dataset),
                       100. * batch_idx / len(valid_loader), loss.item()))
        torch.cuda.empty_cache()