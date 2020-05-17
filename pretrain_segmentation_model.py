import cv2
import segmentation_models_pytorch as smp
import torch

from pytorch_metric_learning import losses
# -- MODEL --
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader


from utils.CubeDataset import CubeDataset
from utils.UNet import Unet
from utils.dataset_utils import visualize_tensor, transform_from_log, visualize, to_np_img
from utils.transformation_utils import get_training_augmentation, get_preprocessing, get_validation_augmentation
import numpy as np

if __name__ == '__main__':

    use_log = True
    out_channels = 2 if use_log else 3
    known_ills = False
    in_channels = out_channels*3 if known_ills else out_channels

    model = Unet(in_channels, pretrain=True)
    model.cuda(0)

    num_workers = 0
    bs = 4

    train_dataset = CubeDataset(datatype='train',
                              transforms=get_training_augmentation(),
                              log_transform=use_log)  # , preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = CubeDataset(datatype='valid',
                              transforms=get_validation_augmentation(),
                              log_transform=use_log)  # , preprocessing=get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

    # model, criterion, optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-2)
    criterion1 = torch.nn.MSELoss()

    from utils.dataset_utils import visualize_tensor, transform_from_log, visualize, to_np_img
    def plot(data, gt, out_a, gs, gt_gs):
        gs = gs[0]
        gt_gs = np.array([[np.array(gt_gs[0]) for i in range(64)] for j in range(64)])
        d = to_np_img(data[0])
        d = cv2.split(d)
        d = np.dstack((d[0], d[1]))
        d = transform_from_log(d, gs)
        gt_img = np.array([[np.array(gt[0].cpu()) for i in range(64)] for j in range(64)])
        gt_img = transform_from_log(gt_img, gt_gs)
        pred_img = np.array([[out_a[0].detach().cpu().numpy() for i in range(64)] for j in range(64)])
        pred_img = transform_from_log(pred_img, gt_gs)
        visualize(d, gt_img, pred_img)


    num_epochs = 1000
    log_interval = 5
    logdir = "./logs/segmentation"
    min_valid_loss = 0
    for epoch in range(num_epochs):
        for batch_idx, (data, gt, gs, gt_gs) in enumerate(train_loader):
            if epoch == 30:
                optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
            if epoch == 100:
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            out = model(data)
            loss = criterion1(out, gt).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 100 != 0 or epoch < 150:
                if batch_idx % 1000 == 0:
                    plot(data, gt, out, gs, gt_gs)
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            torch.cuda.empty_cache()
        cum_loss = 0
        for batch_idx, (data, gt, gs, gt_gs) in enumerate(valid_loader):
            out = model(data)
            loss = criterion1(out, gt)
            cum_loss += loss.mean().detach()
            if epoch % 10 != 0 or epoch < 20:
                if batch_idx % 10 == 0:
                    plot(data, gt, out, gs, gt_gs)
            if batch_idx % log_interval == 0:
                print('Valid Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(valid_loader.dataset),
                           100. * batch_idx / len(valid_loader), loss.item()))
            torch.cuda.empty_cache()
        if epoch == 0:
            min_valid_loss = cum_loss
        if min_valid_loss > cum_loss:
            min_valid_loss = cum_loss
            min_epoch = epoch
            torch.save(model.state_dict(), './models/unet-pretrained-cube')