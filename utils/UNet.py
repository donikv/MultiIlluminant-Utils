import torch
import torch.nn as nn


class Unet(nn.Module):

    def __init__(self, in_channels=2, num_classes=1, use_sigmoid=False, pretrain=False):
        super(Unet, self).__init__()

        self.use_sigmoid = use_sigmoid

        #128*128
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.down1_pool = nn.Sequential(nn.MaxPool2d(kernel_size=8, stride=8))

        #64*64
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),)
        self.down2_pool = nn.Sequential(nn.MaxPool2d(kernel_size=8, stride=8))

        # #32*32
        # self.down3 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),)
        # self.down3_pool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        #
        # # 16*16
        # self.down4 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True))
        # self.down4_pool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        # 512*8*8
        self.center = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),)

        self.pretrain = pretrain
        if self.pretrain:
            self.pretrain_classifier = nn.Sequential(
                nn.Linear(12800, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, in_channels),
            )
            return

        # self.upsample4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'))
        #
        # self.up4 = nn.Sequential(
        #     nn.Conv2d(1024+512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),)
        #
        # self.upsample3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'))
        # self.up3 = nn.Sequential(
        #     nn.Conv2d(512+256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),)

        self.upsample2 = nn.Sequential(nn.Upsample(scale_factor=8, mode='bilinear'))
        self.up2 = nn.Sequential(
            nn.Conv2d(256+128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),)

        self.upsample1 = nn.Sequential(nn.Upsample(scale_factor=8, mode='bilinear'))
        self.up1 = nn.Sequential(
            nn.Conv2d(128+64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            )
        if use_sigmoid:
            self.classifier = nn.Sequential(
                    nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1),
                    nn.Sigmoid(),
                )
        else:
            self.classifier = nn.Sequential(
                    nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1),
                )
    # 128

    def forward(self, img):
        #128*128
        down1 = self.down1(img)
        down1_pool = self.down1_pool(down1)

        #64*64
        down2 = self.down2(down1_pool)
        down2_pool = self.down2_pool(down2)

        # #32*32
        # down3 = self.down3(down2_pool)
        # down3_pool = self.down3_pool(down3)
        # #16*16
        # down4 = self.down4(down3_pool)
        # down4_pool = self.down4_pool(down4)
        #8*8
        center = self.center(down2_pool)
        #8*8

        if self.pretrain:
            center = center.view(-1, self.__num_flat_features(center))
            return self.pretrain_classifier(center)

        # up4 = self.upsample4(center)
        # #16*16
        #
        # up4 = torch.cat((down4,up4), 1)
        # up4 = self.up4(up4)
        #
        # up3 = self.upsample3(up4)
        # up3 = torch.cat((down3,up3), 1)
        # up3 = self.up3(up3)

        up2 = self.upsample2(center)
        up2 = torch.cat((down2,up2), 1)
        up2 = self.up2(up2)

        up1 = self.upsample1(up2)
        up1 = torch.cat((down1,up1), 1)
        up1 = self.up1(up1)

        prob = self.classifier(up1)
        print(prob.size())
        if not self.use_sigmoid:
            prob = prob.clamp(0, 1)

        return prob

    @staticmethod
    def __num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def load_pretrained(self, path):
        dict = torch.load(path)
        self.load_state_dict(dict)