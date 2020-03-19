import torch
from torch import nn
from torch.optim import optimizer

from Models import get_model


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
        return [1, 0] if la > lb else [0, 1], la, lb

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
        self.model = get_model(num_classes=2)

    def forward(self, image):
        return self.model(image)
