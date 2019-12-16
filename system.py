import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from argparse import Namespace
from torch.utils.data import DataLoader, Subset

from models import UNet
from data_utils import TimedImageDataset, prefix_sum


class TimeTransfer(pl.LightningModule):
    def __init__(self, hparams):
        super(TimeTransfer, self).__init__()
        self.hparams = hparams

        # networks
        self.unet = UNet(3, hparams.hidden_dim)

        self.data_dir = r'E:\TimeLapseVDataDownsampled'
        self.split_indices = prefix_sum(hparams.data_split)

        self.example_input_array = torch.zeros((4, 3, 800, 450))

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(Subset(TimedImageDataset(self.data_dir), range(self.split_indices[0])),
                          batch_size=self.hparams.batch_size, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(Subset(TimedImageDataset(self.data_dir), range(self.split_indices[0],
                                                                         self.split_indices[1])),
                          batch_size=self.hparams.batch_size, shuffle=True)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(Subset(TimedImageDataset(self.data_dir), range(self.split_indices[1],
                                                                         self.split_indices[2])),
                          batch_size=self.hparams.batch_size, shuffle=True)


if __name__ == '__main__':
    args = {
        'batch_size': 64,
        'lr': 1e-4,
        'hidden_dim': 16,
        'data_split': [8000, 1000, 1000]
    }
    hparams = Namespace(**args)
    time_transfer = TimeTransfer(hparams)
    trainer = pl.Trainer(gpus=0)
    # trainer.fit(time_transfer)
