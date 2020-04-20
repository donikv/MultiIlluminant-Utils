import torch
from torch import nn
from torch.optim import optimizer
import torch.nn.functional as F
import numpy as np

from Models import get_model
from dataset_utils import get_patches_for_image, combine_patches_into_image


class HypNet(nn.Module):

    def __init__(self, patch_width: int, patch_height: int, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.c1_out = 128
        self.c2_out = 256
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, self.c1_out, (8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(self.c1_out, self.c2_out, (4, 4), stride=2),
            nn.ReLU(),
            nn.Dropout(),
        )

        in_size = self.c2_out * 6 * 6 #TODO make calculations for FCL input size

        self.branch_a = nn.Sequential(
            nn.Linear(in_size, 120),
            nn.ReLU(),
            nn.Linear(120, self.out_channels),
        )
        self.branch_b = nn.Sequential(
            nn.Linear(in_size, 120),
            nn.ReLU(),
            nn.Linear(120, self.out_channels),
        )

        for ap in self.branch_a.parameters():
            ap.data = ap.data.normal_(0.0, 0.5)
        for ap in self.branch_b.parameters():
            ap.data = ap.data.normal_(0.0, 0.5)

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

    @staticmethod
    def get_selection_class(outa, outb, gt, loss):
        la = loss(outa, gt)
        lb = loss(outb, gt)
        return ([1, 0]) if la > lb else ([0, 1]), la, lb

    @staticmethod
    def get_selection_class_per_patch(outa, outb, gt, loss):
        if len(gt.shape) == 1:
            gt = torch.stack([gt for i in range(outa.shape[0])])
        sels = []
        for pa, pb, gtp in zip(outa, outb, gt):
            la = loss(outa, gt)
            lb = loss(outb, gt)
            sels.append(torch.tensor([1, 0]) if la > lb else torch.tensor([0, 1]))
        return sels

    def back_pass(self, outa, outb, gt, optim: optimizer.Optimizer, loss, train_both=False):
        optim.zero_grad()
        la = loss(outa, gt)
        lb = loss(outb, gt)
        if train_both:
            self.branch_a.requires_grad = False
            lb.backward(retain_graph=True)
            self.branch_a.requires_grad = True
            self.branch_b.reqires_grad = False
            la.backward(retain_graph=False)
            self.branch_b.reqires_grad = True
        elif la > lb:
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

    def select(self, hypNet: HypNet, image, gt, patch_height_ratio, patch_width_ratio, loss, is_gt_image=True):
        patches = get_patches_for_image(image, patch_height_ratio, patch_width_ratio)
        if is_gt_image:
            gt = get_patches_for_image(gt, patch_height_ratio, patch_width_ratio).mean(2).mean(2)
        # p_mean = patches.mean(3, keepdim=True).mean(2, keepdim=True)
        # patches = patches - p_mean
        # p_s_mean = p_mean.squeeze().squeeze()
        out_a, out_b = hypNet(patches)
        # out_a += p_s_mean
        # out_b += p_s_mean
        sel = torch.stack(hypNet.get_selection_class_per_patch(out_a, out_b, gt, loss)) # gt - p_mean

        sel_img = combine_patches_into_image(sel, patch_height_ratio, patch_width_ratio).transpose(0, 1).transpose(0, 2)
        _, image_height, image_width = image.shape
        _, sel_img_height, sel_img_width = sel_img.shape
        sel_img_gt = F.pad(sel_img, [0, image_width - sel_img_width, 0, image_height - sel_img_height],
                           'constant').cuda(0)
        return self.model(image.unsqueeze(0)), sel_img_gt.unsqueeze(0)

    def test(self, hypNet: HypNet, image, gt, patch_height_ratio, patch_width_ratio):
        patches = get_patches_for_image(image, patch_height_ratio, patch_width_ratio)
        # p_mean = patches.mean(3, keepdim=True).mean(2, keepdim=True)
        # patches = patches - p_mean
        # p_s_mean = p_mean.squeeze().squeeze()
        out_a, out_b = hypNet(patches)
        # out_a += p_s_mean
        # out_b += p_s_mean
        sel_a = combine_patches_into_image(out_a, patch_height_ratio, patch_width_ratio).transpose(0, 1).transpose(0, 2)
        sel_b = combine_patches_into_image(out_b, patch_height_ratio, patch_width_ratio).transpose(0, 1).transpose(0, 2)
        _, image_height, image_width = image.shape
        _, sel_img_height, sel_img_width = sel_a.shape
        sel_a = F.pad(sel_a, [0, image_width - sel_img_width, 0, image_height - sel_img_height],
                      'constant').cuda(0)
        sel_b = F.pad(sel_b, [0, image_width - sel_img_width, 0, image_height - sel_img_height],
                      'constant').cuda(0)
        selection_map = self.model(image.unsqueeze(0))[0][0]

        return SelUnet.get_selection_from_map(sel_a, sel_b, selection_map)

    @staticmethod
    def get_selection_from_map(sel_a: torch.Tensor, sel_b: torch.Tensor, selection_map: torch.Tensor):
        sel_a = sel_a.transpose(2, 0).transpose(1, 0)
        sel_b = sel_b.transpose(2, 0).transpose(1, 0)
        selection_map = selection_map.transpose(2, 0).transpose(1, 0)

        def select(i, j, sel_map, sel_a, sel_b):
            pix = sel_map[i][j]
            return sel_a[i][j] if pix[1] > pix[0] else sel_b[i][j]

        final = [torch.stack([select(i, j, selection_map, sel_a, sel_b) for j in range(selection_map.shape[1])]) for i in
                 range(selection_map.shape[0])]
        return torch.stack(final).transpose(0, 1).transpose(0, 2)
