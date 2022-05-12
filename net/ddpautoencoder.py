import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms

class DDPAutoEncoder(pl.LightningModule):

    def __init__(self, root, batchsize, hidden_dim: int = 64):
        super().__init__()
        self.root = root
        self.batchsize = batchsize
        self.loss = F.mse_loss
        self.encoder = nn.Sequential(
            nn.Conv2d(3,8,kernel_size=5, padding=2), 
            nn.ReLU(), 
            nn.MaxPool2d(4,4),
            nn.Conv2d(8,16,kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(5,5),
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16,8,kernel_size=5,stride = 5),
            nn.ReLU(),
            nn.ConvTranspose2d(8,3,kernel_size=5,stride = 4, padding=1, output_padding=1),
            nn.ReLU()
            )

    def forward(self, x):
        embedding = self.encoder(x)
        output = self.decoder(embedding)
        return [embedding,output]

    def training_step(self, batch, batch_idx):
        x,y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss(x_hat, x)
        self.log("training_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss(x_hat, x)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x,y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss(x_hat, x)
        self.log("test_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch
        z = self.encoder(x)
        return self.decoder(z)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def setup(self, stage):
        data_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.train_data = ImageFolder(self.root,transform=data_transform)
        self.val_data = ImageFolder(self.root,transform=data_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data,batch_size = self.batchsize,num_workers=24)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,batch_size = self.batchsize,num_workers=24)