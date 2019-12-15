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
            nn.Conv2d(n_channels, hidden_dim, kernel_size=(4, 3), padding=(2, 1), padding_mode='same'),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=(4, 3), padding=(1, 1), padding_mode='same'),
            nn.ReLU(True),
            nn.InstanceNorm2d(hidden_dim * 2))

        self.down2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=(8, 6), stride=(4, 3), padding=(4, 2)),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=(4, 3), padding=(1, 1)),
            nn.ReLU(True),
            nn.InstanceNorm2d(hidden_dim * 4))

        self.down3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=(8, 6), stride=(4, 3), padding=(4, 2)),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=(4, 3), padding=(1, 1)),
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

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=(4, 3), mode='bilinear', align_corners=True),
                                 nn.Conv2d(hidden_dim * 16, hidden_dim * 4, kernel_size=(4, 3), padding=(2, 1)),
                                 nn.ReLU(True),
                                 nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=(4, 3), padding=(1, 1)),
                                 nn.ReLU(True),
                                 nn.InstanceNorm2d(hidden_dim * 4))

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=(4, 3), mode='bilinear', align_corners=True),
                                 nn.Conv2d(hidden_dim * 8, hidden_dim * 2, kernel_size=(4, 3), padding=(2, 1)),
                                 nn.ReLU(True),
                                 nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=(4, 3), padding=(1, 1)),
                                 nn.ReLU(True),
                                 nn.InstanceNorm2d(hidden_dim * 2))

        self.up4 = nn.Sequential(nn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=(4, 3), padding=(2, 1)),
                                 nn.ReLU(True),
                                 nn.Conv2d(hidden_dim * 2, n_channels, kernel_size=(4, 3), padding=(1, 1)))

    def combine(self, x1, x2):
        return torch.cat([x1, x2], dim=1)

    def encode_time(self, t):
        angle = 2 * np.pi / 24 * t
        angle = angle.unsqueeze(-1)
        return torch.cat([angle.cos(), angle.sin()], dim=1)

    def forward(self, x, t):
        # downsampling
        print(x.shape)
        x1 = self.down1(x)
        print(x1.shape)
        x2 = self.down2(x1)
        print(x2.shape)
        x3 = self.down3(x2)
        print(x3.shape)
        x4 = self.down4(x3)
        print(x4.shape)

        # x = x4.view((x4.size(0), -1))
        # print(x.shape)
        encoded_t = self.encode_time(t)
        time_features = encoded_t.unsqueeze(0).repeat(1, 1, 25*25).view(-1, 2, 25, 25)
        # upsampling
        x = self.up1(self.combine(x4, time_features))
        print(x.shape)
        x = self.up2(self.combine(x, x3))
        print(x.shape)
        x = self.up3(self.combine(x, x2))
        print(x.shape)
        x = self.up4(self.combine(x, x1))
        print(x.shape)
        # out = self.out_conv(x)
        # print(out.shape)
        return x


if __name__ == '__main__':
    unet = UNet(3, 16)
    example = torch.zeros((4, 3, 800, 450))
    t = torch.zeros(4)
    out = unet(example, t)
