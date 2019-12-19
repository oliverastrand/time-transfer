import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from argparse import Namespace
from torch.utils.data import DataLoader, Subset
import torchvision
import os

from models import UNet, HyperUNet
from data_utils import TimedImageDataset, prefix_sum
from losses import PerceptualLoss


class TimeTransfer(pl.LightningModule):
    def __init__(self, hparams):
        super(TimeTransfer, self).__init__()
        self.hparams = hparams

        # networks
        self.unet = HyperUNet(3, hparams.hidden_dim)

        self.data_dir = os.path.expanduser(hparams.data_dir)

        self.split_indices = prefix_sum(hparams.data_split)

        self.perceptual_loss = PerceptualLoss()

        # self.example_input_array = torch.zeros((4, 3, 450, 800)), torch.tensor([3, 6, 12, 21])

    def forward(self, x, t):
        t = t * torch.ones(x.shape[0]).to(x.device)
        return self.unet(x, t)

    def get_time_batch(self, batch, t):
        x = batch[t]
        return x

    def training_step(self, batch, batch_nb):
        # REQUIRED
        source_hour = 12
        target_hour = 18
        x = self.get_time_batch(batch, source_hour)
        y = self.get_time_batch(batch, target_hour)
        y_hat = self.forward(x, target_hour)
        # mse_loss = F.mse_loss(y_hat, y)
        loss = self.perceptual_loss(y_hat, y)
        # style_loss = self.style_loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss,
                            # 'mse_loss': mse_loss,
                            # 'perceptual_loss': content_loss + style_loss,
                            # 'content_loss': content_loss,
                            # 'style_loss': style_loss
                            }
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        source_hour = 12
        target_hour = 18
        x = self.get_time_batch(batch, source_hour)
        y = self.get_time_batch(batch, target_hour)
        y_hat = self.forward(x, target_hour)
        loss = self.perceptual_loss(y_hat, y)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        source_hour = 12
        target_hour = 18
        x = self.get_time_batch(batch, source_hour)
        y = self.get_time_batch(batch, target_hour)
        y_hat = self.forward(x, target_hour)
        loss = F.mse_loss(y_hat, y)
        return {'test_loss': loss}

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def on_epoch_end(self):
        # log sampled images
        dataset = self.test_dataloader()[0].dataset
        samples = dataset[:self.hparams.n_samples]
        source_hour = 12
        target_hour = 18

        device = next(self.unet.parameters()).device
        x = self.get_time_batch(samples, source_hour).to(device)
        y = self.get_time_batch(samples, target_hour).to(device)
        y_hat = self.forward(x, target_hour)
        grid = torchvision.utils.make_grid(torch.stack([x, y_hat, y], dim=1).view(-1, 3, 450, 800),
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
                          batch_size=self.hparams.batch_size, shuffle=True, num_workers=2)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(Subset(TimedImageDataset(self.data_dir), range(self.split_indices[0],
                                                                         self.split_indices[1])),
                          batch_size=self.hparams.batch_size, shuffle=True, num_workers=2)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(Subset(TimedImageDataset(self.data_dir), range(self.split_indices[1],
                                                                         self.split_indices[2])),
                          batch_size=self.hparams.batch_size, shuffle=True, num_workers=2)


if __name__ == '__main__':
    args = {
        'batch_size': 8,
        'lr': 0.01,
        'hidden_dim': 4,
        'data_split': [5000, 1000, 1000],
        'n_samples': 10,
        'data_dir': r'~/E/TimeLapseVDataDownsampled'
    }
    hparams = Namespace(**args)
    time_transfer = TimeTransfer.load_from_checkpoint('logs/hyper_unet_logs_12_to_21/checkpoints/_ckpt_epoch_4.ckpt')

    from pytorch_lightning.callbacks import ModelCheckpoint

    save_path = 'logs/hyper_unet_logs_12_to_18_from_21'
    # DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=f'{save_path}/checkpoints',
        save_best_only=False,
        verbose=True,
    )

    # most basic trainer, uses good defaults (1 gpu)
    trainer = pl.Trainer(gpus=1,
                         default_save_path=f'{save_path}',
                         checkpoint_callback=checkpoint_callback,
                         early_stop_callback=None,
                         max_nb_epochs=50)
    trainer.fit(time_transfer)
    # trainer.test(time_transfer)
