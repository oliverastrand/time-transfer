import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from data_utils import TimedImageDataset
from models import UNet


class DumbSystem(pl.LightningModule):
    def __init__(self):
        super(DumbSystem, self).__init__()
        # not the best model...
        self.encoder = DumbEncoder()
        self.decoder = DumbDecoder()

        self.batch_size = 2

    def training_step(self, batch, batch_nb):
        # REQUIRED

        loss = 0
        for time_string, images in batch.items():
            time = int(time_string)
            time_vec = (torch.ones(images.shape[0]).to('cuda') * time).unsqueeze(1)

            enc = self.encoder(images)
            enc = torch.cat((time_vec, enc), dim=1)

            dec = self.decoder(enc)

            loss += torch.sum((dec - images) ** 2)

        return {'loss': loss}

    def configure_optimizers(self):
        # REQUIRED
        return torch.optim.Adam(self.parameters(), lr=0.02)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(TimedImageDataset('./data/dataSamples'), batch_size=self.batch_size)
