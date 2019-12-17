import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from argparse import Namespace
from torch.utils.data import DataLoader, Subset
import torchvision

from models import UNet
from data_utils import TimedImageDataset, prefix_sum


class TimeTransfer(pl.LightningModule):
    def __init__(self, hparams):
        super(TimeTransfer, self).__init__()
        self.hparams = hparams

        # networks
        self.unet = UNet(3, hparams.hidden_dim)

        self.data_dir = r'/Users/ruoyi/Downloads/TimeLapseVDataDownsampled'

        self.split_indices = prefix_sum(hparams.data_split)

        self.example_input_array = torch.zeros((4, 3, 800, 450))

    def forward(self, x):
        return self.unet(x)

    def get_time_batch(self, batch, t):
        x = batch[t]
        return x

    def training_step(self, batch, batch_nb):
        # REQUIRED
        source_hour = 12
        target_hour = 12
        x = self.get_time_batch(batch, source_hour)
        y = self.get_time_batch(batch, target_hour)
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        source_hour = 12
        target_hour = 12
        x = self.get_time_batch(batch, source_hour)
        y = self.get_time_batch(batch, target_hour)
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        source_hour = 12
        target_hour = 12
        x = self.get_time_batch(batch, source_hour)
        y = self.get_time_batch(batch, target_hour)
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        return {'test_loss': loss}

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def on_epoch_end(self):
        # log sampled images
        dataset = self.test_dataloader().dataset
        samples = dataset[:10]
        source_hour = 12
        target_hour = 12
        x = self.get_time_batch(samples, source_hour)
        y = self.get_time_batch(samples, target_hour)
        y_hat = self.forward(x)
        grid = torchvision.utils.make_grid(torch.stack([x, y_hat, y], dim=1).view(-1, 3, 800, 450),
                                           nrow=3)
        self.logger.experiment.add_image(f'samples', grid, self.current_epoch)

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
