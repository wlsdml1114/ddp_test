import torch
import pytorch_lightning as pl
from typing import Optional
from torchvision.datasets import ImageFolder
from torchvision import transforms

class AutoEncoderDataLoader(pl.LightningDataModule):
    def __init__(self, root, batchsize = 16,num_workers=24):
        super().__init__()
        self.root = root
        self.batchsize = batchsize
        self.num_workers = num_workers
        
    def setup(self, stage: Optional[str] = None):
        data_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.train_data = ImageFolder(self.root,transform=data_transform)
        self.val_data = ImageFolder(self.root,transform=data_transform)
        print('dataset length :', len(self.train_data))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data,batch_size = self.batchsize,num_workers=self.num_workers)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,batch_size = self.batchsize,num_workers=self.num_workers)