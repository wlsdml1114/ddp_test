import os
import pytorch_lightning as pl
import torch
from torchvision.datasets import ImageFolder

class IsedolDataset(pl.LightningDataModule):
    def __init__(self, root, batchsize = 16, num=0):
        self.root = root
        self.batchsize = batchsize

    def prepare_data(self):
        return super().prepare_data()

    def setup(self, stage):
        self.train_data = ImageFolder(os.path.join(self.root,'Images'))
        self.val_data = ImageFolder(os.path.join(self.root,'Images'))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data,batch_size = self.batchsize)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,batch_size = self.batchsize)