import torch
import pytorch_lightning as pl
from argparse import Namespace
from models import UNet


class TimeTransfer(pl.LightningModule):
    def __init__(self, hparams):
        super(TimeTransfer, self).__init__()
        self.hparams = hparams

        # networks
        self.unet = UNet(3)

        self.example_input_array = torch.zeros((4, 3, 800, 450))

    def forward(self, x):
        return self.unet(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == '__main__':
    args = {
        'batch_size': 64,
        'lr': 1e-5,
        'beta_1': 0.5,
        'beta_2': 0.9,
        'latent_dim': 100,
        'model_dim': 128,
        'gp_lambda': 10,
        'disc_per_gen': 8
    }
    hparams = Namespace(**args)
    time_transfer = TimeTransfer(hparams)
    trainer = pl.Trainer(gpus=0)
    # trainer.fit(time_transfer)
