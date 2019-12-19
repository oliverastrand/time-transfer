import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UNet(nn.Module):
    def __init__(self, n_channels, hidden_dim):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim

        self.down1 = nn.Sequential(
            nn.Conv2d(n_channels, hidden_dim, kernel_size=(3, 4), padding=(1, 2), padding_mode='same'),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=(3, 4), padding=(1, 1), padding_mode='same'),
            nn.ReLU(True),
            nn.InstanceNorm2d(hidden_dim * 2))

        self.down2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=(6, 8), stride=(3, 4), padding=(2, 4)),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=(3, 4), padding=(1, 1)),
            nn.ReLU(True),
            nn.InstanceNorm2d(hidden_dim * 4))

        self.down3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=(6, 8), stride=(3, 4), padding=(2, 4)),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=(3, 4), padding=(1, 1)),
            nn.ReLU(True),
            nn.InstanceNorm2d(hidden_dim * 8))

        self.down4 = nn.Sequential(nn.Conv2d(hidden_dim * 8, hidden_dim * 16, kernel_size=(4, 4), stride=2),
                                   nn.ReLU(True),
                                   nn.Conv2d(hidden_dim * 16, hidden_dim * 16, kernel_size=(4, 4), padding=2),
                                   nn.ReLU(True),
                                   nn.InstanceNorm2d(hidden_dim * 16))

        # self.dense = nn.Linear(hidden_dim * 16 * 25 * 25 + 2, hidden_dim * 16 * 25 * 25)

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(hidden_dim * 16 + 2, hidden_dim * 8, kernel_size=(4, 4), padding=2),
                                 nn.ReLU(True),
                                 nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=(4, 4), padding=1),
                                 nn.ReLU(True),
                                 nn.InstanceNorm2d(hidden_dim * 8))

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=(3, 4), mode='bilinear', align_corners=True),
                                 nn.Conv2d(hidden_dim * 16 + 2, hidden_dim * 4, kernel_size=(3, 4), padding=(1, 2)),
                                 nn.ReLU(True),
                                 nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=(3, 4), padding=(1, 1)),
                                 nn.ReLU(True),
                                 nn.InstanceNorm2d(hidden_dim * 4))

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=(3, 4), mode='bilinear', align_corners=True),
                                 nn.Conv2d(hidden_dim * 8 + 2, hidden_dim * 2, kernel_size=(3, 4), padding=(1, 2)),
                                 nn.ReLU(True),
                                 nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=(3, 4), padding=(1, 1)),
                                 nn.ReLU(True),
                                 nn.InstanceNorm2d(hidden_dim * 2))

        self.up4 = nn.Sequential(nn.Conv2d(hidden_dim * 4 + 2, hidden_dim * 2, kernel_size=(3, 4), padding=(1, 2)),
                                 nn.ReLU(True),
                                 nn.Conv2d(hidden_dim * 2, n_channels, kernel_size=(3, 4), padding=(1, 1)))

    def combine(self, *x):
        return torch.cat(x, dim=1)

    def encode_time(self, t):
        angle = 2 * np.pi / 24 * t
        angle = angle.unsqueeze(-1)
        return torch.cat([angle.cos(), angle.sin()], dim=1)

    def time_feature_map(self, encoded_time, shape):
        return encoded_time.unsqueeze(0).repeat(1, 1, torch.prod(torch.tensor(shape))).view(-1, 2, *shape)

    def forward(self, x, t):
        # downsampling
        # print(x.shape)
        x1 = self.down1(x)
        # print(x1.shape)
        x2 = self.down2(x1)
        # print(x2.shape)
        x3 = self.down3(x2)
        # print(x3.shape)
        x4 = self.down4(x3)
        # print(x4.shape)

        # x = x4.view((x4.size(0), -1))
        # print(x.shape)
        encoded_t = self.encode_time(t)
        # time_features = self.time_feature_map(encoded_t, (25, 25))
        # upsampling
        x = self.up1(self.combine(x4, self.time_feature_map(encoded_t, (25, 25))))
        # print(x.shape)
        x = self.up2(self.combine(x, x3, self.time_feature_map(encoded_t, (50, 50))))
        # print(x.shape)
        x = self.up3(self.combine(x, x2, self.time_feature_map(encoded_t, (150, 200))))
        # print(x.shape)
        x = self.up4(self.combine(x, x1, self.time_feature_map(encoded_t, (450, 800))))
        # print(x.shape)
        # out = self.out_conv(x)
        # print(out.shape)
        return x


class HyperUNet(nn.Module):
    def __init__(self, n_channels, hidden_dim):
        super(HyperUNet, self).__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim

        self.down1 = nn.Sequential(
            nn.Conv2d(n_channels, hidden_dim, kernel_size=(3, 4), padding=(1, 2), padding_mode='same'),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=(3, 4), padding=(1, 1), padding_mode='same'),
            nn.ReLU(True),
            nn.InstanceNorm2d(hidden_dim * 2))

        self.down2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=(6, 8), stride=(3, 4), padding=(2, 4)),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=(3, 4), padding=(1, 1)),
            nn.ReLU(True),
            nn.InstanceNorm2d(hidden_dim * 4))

        self.down3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=(6, 8), stride=(3, 4), padding=(2, 4)),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=(3, 4), padding=(1, 1)),
            nn.ReLU(True),
            nn.InstanceNorm2d(hidden_dim * 8))

        self.down4 = nn.Sequential(nn.Conv2d(hidden_dim * 8, hidden_dim * 16, kernel_size=(4, 4), stride=2),
                                   nn.ReLU(True),
                                   nn.Conv2d(hidden_dim * 16, hidden_dim * 16, kernel_size=(4, 4), padding=2),
                                   nn.ReLU(True),
                                   nn.InstanceNorm2d(hidden_dim * 16))

        self.time_conv = nn.Linear(2, 3 * 3 * (hidden_dim * 16) * (hidden_dim * 16))

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(hidden_dim * 16, hidden_dim * 8, kernel_size=(4, 4), padding=2),
                                 nn.ReLU(True),
                                 nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=(4, 4), padding=1),
                                 nn.ReLU(True),
                                 nn.InstanceNorm2d(hidden_dim * 8))

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=(3, 4), mode='bilinear', align_corners=True),
                                 nn.Conv2d(hidden_dim * 16, hidden_dim * 4, kernel_size=(3, 4), padding=(1, 2)),
                                 nn.ReLU(True),
                                 nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=(3, 4), padding=(1, 1)),
                                 nn.ReLU(True),
                                 nn.InstanceNorm2d(hidden_dim * 4))

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=(3, 4), mode='bilinear', align_corners=True),
                                 nn.Conv2d(hidden_dim * 8, hidden_dim * 2, kernel_size=(3, 4), padding=(1, 2)),
                                 nn.ReLU(True),
                                 nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=(3, 4), padding=(1, 1)),
                                 nn.ReLU(True),
                                 nn.InstanceNorm2d(hidden_dim * 2))

        self.up4 = nn.Sequential(nn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=(3, 4), padding=(1, 2)),
                                 nn.ReLU(True),
                                 nn.Conv2d(hidden_dim * 2, n_channels, kernel_size=(3, 4), padding=(1, 1)))

    def combine(self, *x):
        return torch.cat(x, dim=1)

    def encode_time(self, t):
        angle = 2 * np.pi / 24 * t
        angle = angle.unsqueeze(-1)
        return torch.cat([angle.cos(), angle.sin()], dim=1)

    def time_feature_map(self, encoded_time, shape):
        return encoded_time.unsqueeze(0).repeat(1, 1, torch.prod(torch.tensor(shape))).view(-1, 2, *shape)

    def forward(self, x, t):
        # downsampling
        # print(x.shape)
        x1 = self.down1(x)
        # print(x1.shape)
        x2 = self.down2(x1)
        # print(x2.shape)
        x3 = self.down3(x2)
        # print(x3.shape)
        x4 = self.down4(x3)
        # print(x4.shape)

        # x = x4.view((x4.size(0), -1))
        # print(x.shape)
        encoded_t = self.encode_time(t)
        batch_size = x4.size(0)
        n_channels = x4.size(1)
        # fixme: this assumes all entries in the same batch uses the same time
        # fixme: due to pytorch quirks on broadcasting conv2d
        time_conv = self.time_conv(encoded_t).view(batch_size, n_channels, n_channels, 3, 3)[0]
        after_conv = F.conv2d(x4, time_conv, padding=1)
        # upsampling
        x = self.up1(after_conv)
        # print(x.shape)
        x = self.up2(self.combine(x, x3))
        # print(x.shape)
        x = self.up3(self.combine(x, x2))
        # print(x.shape)
        x = self.up4(self.combine(x, x1))
        # print(x.shape)
        # out = self.out_conv(x)
        # print(out.shape)
        return x


if __name__ == '__main__':
    unet = UNet(3, 16)
    example = torch.zeros((4, 3, 450, 800))
    t = torch.zeros(4)
    out = unet(example, t)
