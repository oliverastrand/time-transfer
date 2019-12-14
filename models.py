import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownNet(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpNet(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


""" Full assembly of the parts to form the complete network """


class UNet(nn.Module):
    def __init__(self, n_channels, hidden_dim):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim

        self.down1 = nn.Sequential(
            nn.Conv2d(n_channels, hidden_dim, kernel_size=(4, 3), padding=(2, 1), padding_mode='same'),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=(4, 3), padding=(1, 1), padding_mode='same'),
            nn.InstanceNorm2d(hidden_dim * 2))

        self.down2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=(8, 6), stride=(4, 3), padding=(4, 2)),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=(4, 3), padding=(1, 1)),
            nn.InstanceNorm2d(hidden_dim * 4))

        self.down3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=(8, 6), stride=(4, 3), padding=(4, 2)),
            nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=(4, 3), padding=(1, 1)),
            nn.InstanceNorm2d(hidden_dim * 8))

        self.down4 = nn.Sequential(nn.Conv2d(hidden_dim * 8, hidden_dim * 16, kernel_size=(4, 4), stride=2),
                                   nn.Conv2d(hidden_dim * 16, hidden_dim * 16, kernel_size=(4, 4), padding=2),
                                   nn.InstanceNorm2d(hidden_dim * 16))

        # self.down5 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(4, 4), stride=2),
        #                            nn.Conv2d(512, 512, kernel_size=(4, 4)),
        #                            nn.InstanceNorm2d(512))

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(hidden_dim * 16, hidden_dim * 8, kernel_size=(4, 4), padding=2),
                                 nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=(4, 4), padding=1),
                                 nn.InstanceNorm2d(hidden_dim * 8))

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=(8, 6), stride=(4, 3), padding=(2, 1)),
            # nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=(4, 3), padding=(1, 1)),
        )

        self.up3 = nn.Sequential(nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=(8, 6), stride=(4, 3)))

        self.up4 = nn.Sequential(nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=(4, 4), padding=(2, 2)))

        self.out_conv = nn.Sequential(nn.Conv2d(hidden_dim, n_channels, kernel_size=(4, 3)))

    def forward(self, x):
        print(x.shape)
        x1 = self.down1(x)
        print(x1.shape)
        x2 = self.down2(x1)
        print(x2.shape)
        x3 = self.down3(x2)
        print(x3.shape)
        x4 = self.down4(x3)
        print(x4.shape)
        x = self.up1(x4)
        print(x.shape)
        x = self.up2(x)
        print(x.shape)
        x = self.up3(x)
        print(x.shape)
        x = self.up4(x)
        print(x.shape)
        out = self.out_conv(x)
        print(out.shape)
        return out


if __name__ == '__main__':
    unet = UNet(3, 32)
    example = torch.zeros((4, 3, 800, 450))
    out = unet(example)
