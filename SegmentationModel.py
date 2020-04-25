import cv2
import segmentation_models_pytorch as smp
import torch

# -- MODEL --
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader

from Dataset import MIDataset, MIPatchedDataset
from Losses import BCEDiceLoss
from Models import get_model, get_custom_model
from dataset_utils import visualize_tensor, to_np_img, transform_from_log, visualize, mask_to_image
from transformation_utils import get_training_augmentation, get_preprocessing, get_validation_augmentation
import numpy as np


def plot(data, gs, mask, p_mask, use_log, custom_transform=lambda x: x):
    d = to_np_img(data[0])
    if use_log:
        d = cv2.split(d)
        d = np.dstack((d[0], d[1]))
        gs = gs[0]
        d = transform_from_log(d, gs)
    d = d.astype(int)
    mask = mask_to_image(to_np_img(mask[0]))
    p_mask = custom_transform(mask_to_image(to_np_img(p_mask[0])))
    visualize(d, p_mask, mask=mask)

if __name__ == '__main__':

    model, preprocessing_fn = get_model(num_classes=1, use_sigmoid=False)
    model = get_custom_model(num_classes=1, use_sigmoid=False)
    # model.load_pretrained('./models/unet-pretrained-115')
    # dict = torch.load('./models/unet-efficientnet-b0-gt-best-valid-cube3-custom')
    # model.load_state_dict(dict)
    num_workers = 0
    bs = 2
    use_mask = False
    use_log = True
    use_corrected = True
    dataset = 'cube'
    folder = 'dataset_relighted/complex2'
    folder_valid = 'dataset_relighted/complex2/valid'
    train_dataset = MIDataset(folder=folder, datatype='train', dataset=dataset,
                              transforms=get_training_augmentation(), use_mask=use_mask, log_transform=use_log, use_corrected=use_corrected)  # , preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = MIDataset(folder=folder_valid, datatype='valid', dataset=dataset,
                              transforms=get_validation_augmentation(), use_mask=use_mask, log_transform=use_log, use_corrected=use_corrected)  # , preprocessing=get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    num_epochs = 1000
    log_interval = 5
    logdir = "./logs/segmentation"

    # model, criterion, optimizer
    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': 1e-2},
        {'params': model.encoder.parameters(), 'lr': 1e-3},
    ])
    # optimizer = torch.optim.Adam(model.parameters(), lr=7e-2)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
    criterion1 = smp.utils.losses.DiceLoss(eps=1.)
    criterion2 = BCEDiceLoss()

    # -- TRAINING --
    min_valid_loss = 0
    min_epoch = 0
    for epoch in range(num_epochs):
        if epoch == 50:
            optimizer = torch.optim.Adam([
                {'params': model.decoder.parameters(), 'lr': 8e-3},
                {'params': model.encoder.parameters(), 'lr': 1e-3},
            ])
        if epoch == 100:
            optimizer = torch.optim.Adam([
                {'params': model.decoder.parameters(), 'lr': 5e-3},
                {'params': model.encoder.parameters(), 'lr': 1e-3},
            ])
        for batch_idx, (data, mask, gt) in enumerate(train_loader):
            data, gs = data
            p_mask = model(data)
            optimizer.zero_grad()
            loss = criterion2(p_mask, mask).mean()
            #         loss += criterion2(label, )
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            torch.cuda.empty_cache()
        cum_loss = 0
        for batch_idx, (data, mask, gt) in enumerate(valid_loader):
            data, gs = data
            p_mask = model(data)

            optimizer.zero_grad()
            loss = criterion2(p_mask, mask).mean().detach()
            cum_loss += loss
            #         loss += criterion2(label, )
            if batch_idx % log_interval == 0:
                print('Valid Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(valid_loader.dataset),
                           100. * batch_idx / len(valid_loader), loss.item()))
                if epoch % 20 == 0:
                    plot(data, gs, mask, p_mask, use_log)
            torch.cuda.empty_cache()
        print('Valid Epoch: {} \tCumulative loss {} \tMinimum loss {}'.format(
            epoch, cum_loss, min_valid_loss))
        if epoch == 0:
            min_valid_loss = cum_loss
        if min_valid_loss > cum_loss:
            min_valid_loss = cum_loss
            min_epoch = epoch
            torch.save(model.state_dict(), './models/unet-efficientnet-b0-gt-best-valid-cube3-custom2')
    print('Valid Epoch: {} \tMinimum loss {}'.format(
        min_epoch, min_valid_loss))
