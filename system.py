import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from argparse import Namespace
from torch.utils.data import DataLoader, Subset
import torchvision
import os

from models import UNet
from data_utils import TimedImageDataset, prefix_sum

from losses import PerceptualLoss

class TimeTransfer(pl.LightningModule):
    def __init__(self, hparams):
        super(TimeTransfer, self).__init__()
        self.hparams = hparams

        # networks
        self.unet = UNet(3, hparams.hidden_dim)

        self.data_dir = os.path.expanduser(hparams.data_dir)

        self.split_indices = prefix_sum(hparams.data_split)

        self.device = torch.device('cuda' if hparams.gpus > 0 else 'cpu')
        self.criteria = PerceptualLoss().to(self.device)

        # self.example_input_array = torch.zeros((4, 3, 450, 800)), torch.tensor([3, 6, 12, 21])
        #self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=self.hparams.lr) 

    def forward(self, x, t):
        t = t * torch.ones(x.shape[0]).to(x.device)
        return self.unet(x, t)

    def get_time_batch(self, batch, t):
        x = batch[t]
        return x

    def training_step(self, batch, batch_nb):
        # REQUIRED
        source_hour = torch.randint(0, 23, (1,)).item()
        target_hour = torch.randint(0, 23, (1,)).item()
        x = self.get_time_batch(batch, source_hour)
        y = self.get_time_batch(batch, target_hour)
        y_hat = self.forward(x, target_hour)
        loss = self.criteria(y_hat, y)
        #loss = F.mse_loss(y_hat, y)
        #print(loss - loss_test)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        source_hour = torch.randint(0, 23, (1,)).item()
        target_hour = torch.randint(0, 23, (1,)).item()
        x = self.get_time_batch(batch, source_hour)
        y = self.get_time_batch(batch, target_hour)
        y_hat = self.forward(x, target_hour)
        loss = F.mse_loss(y_hat, y)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        source_hour = torch.randint(0, 23, (1,)).item()
        target_hour = source_hour
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
        source_hour = torch.randint(0, 23, (1,)).item()
        target_hour = torch.randint(0, 23, (1,)).item()
        x = self.get_time_batch(samples, source_hour).to(self.device)
        y = self.get_time_batch(samples, target_hour).to(self.device)
        y_hat = self.forward(x, target_hour)
        grid = torchvision.utils.make_grid(torch.stack([x, y_hat, y], dim=1).view(-1, 3, 450, 800),
                                           nrow=3)
        self.logger.experiment.add_image(f'samples', grid, self.current_epoch)

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=self.hparams.lr) 
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 0.5 if x in [3, 7, 12] else 1, last_epoch=-1)
        return [optimizer], [scheduler] 

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(Subset(TimedImageDataset(self.data_dir), range(self.split_indices[0])),
                          batch_size=self.hparams.batch_size, shuffle=True, num_workers=3)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(Subset(TimedImageDataset(self.data_dir), range(self.split_indices[0],
                                                                         self.split_indices[1])),
                          batch_size=self.hparams.batch_size, shuffle=True, num_workers=3)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(Subset(TimedImageDataset(self.data_dir), range(self.split_indices[1],
                                                                         self.split_indices[2])),
                          batch_size=self.hparams.batch_size, shuffle=True, num_workers=3)


if __name__ == '__main__':
    args = {
        'batch_size': 16,
        'lr': 1e-4,
        'hidden_dim': 4,
        'data_split': [4000, 1000, 1000],
        'n_samples': 10,
        'data_dir': r'~/projects/time-transfer/data/TimeLapseVDataDownsampled',
        'gpus': 0,
    }
    hparams = Namespace(**args)
    time_transfer = TimeTransfer(hparams)
    trainer = pl.Trainer(gpus=hparams.gpus, early_stop_callback=None, max_nb_epochs=18)
    trainer.fit(time_transfer)
    # trainer.test(time_transfer)
