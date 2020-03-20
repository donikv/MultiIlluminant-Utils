import torch
from torch import nn
from torch.optim import optimizer
import torch.nn.functional as F

from Models import get_model
from dataset_utils import get_patches_for_image, combine_patches_into_image


class HypNet(nn.Module):

    def __init__(self, patch_width: int, patch_height: int, in_channels: int = 3):
        super().__init__()
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.in_channels = in_channels
        self.c1_out = 128
        self.c2_out = 256
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, self.c1_out, (8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(self.c1_out, self.c2_out, (4, 4), stride=2),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.branch_a = nn.Sequential(
            nn.Linear(self.c2_out * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 3),
        )
        self.branch_b = nn.Sequential(
            nn.Linear(self.c2_out * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 3),
        )


    def __num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, patch):
        cnn_out = self.cnn(patch)
        cnn_out = cnn_out.view(-1, self.__num_flat_features(cnn_out))
        a_out = self.branch_a(cnn_out)
        b_out = self.branch_b(cnn_out)
        return a_out, b_out

    def get_selection_class(self, outa, outb, gt, loss):
        la = loss(outa, gt)
        lb = loss(outb, gt)
        return ([1, 0]) if la > lb else ([0, 1]), la, lb

    def get_selection_class_per_patch(self, outa, outb, gt, loss):
        sels = []
        for pa, pb, gtp in zip(outa, outb, gt):
            la = loss(outa, gt)
            lb = loss(outb, gt)
            sels.append(torch.tensor([1, 0]) if la > lb else torch.tensor([0, 1]))
        return sels

    def back_pass(self, outa, outb, gt, optim: optimizer.Optimizer, loss):
        optim.zero_grad()
        la = loss(outa, gt)
        lb = loss(outb, gt)
        if la > lb:
            self.branch_a.requires_grad = False
            lb.backward()
            self.branch_a.requires_grad = True
        else:
            self.branch_b.reqires_grad = False
            la.backward()
            self.branch_b.reqires_grad = True
        optim.step()
        return [1, 0] if la > lb else [0, 1], la, lb


class SelUnet(nn.Module):

    def __init__(self):
        super().__init__()
        self.model, _ = get_model(num_classes=2)
        self.model.cuda(0)

    def forward(self, image):
        return self.model(image)

    def select(self, hypNet: HypNet, image, gt, patch_height_ratio, patch_width_ratio, loss):
        patches = get_patches_for_image(image, patch_height_ratio, patch_width_ratio)
        gt = get_patches_for_image(gt, patch_height_ratio, patch_width_ratio).mean(2).mean(2)
        out_a, out_b = hypNet(patches)
        sel = torch.stack(hypNet.get_selection_class_per_patch(out_a, out_b, gt, loss))

        sel_img = combine_patches_into_image(sel, patch_height_ratio, patch_width_ratio).transpose(0, 1).transpose(0, 2)
        _, image_height, image_width = image.shape
        _, sel_img_height, sel_img_width = sel_img.shape
        sel_img_gt = F.pad(sel_img, [0, image_width - sel_img_width, 0, image_height-sel_img_height], 'constant').cuda(0)
        return self.model(image.unsqueeze(0)), sel_img_gt.unsqueeze(0)
