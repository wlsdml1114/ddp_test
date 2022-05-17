import torch
import pytorch_lightning as pl
from ..dataset.maskrcnndataset import MaskRCNNDataset
from torchvision.transforms import functional as F
from typing import Tuple, Dict, Optional
from torch import nn, Tensor

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target

class ToTensor(nn.Module):
    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.to_tensor(image)
        return image, target

class MaskRCNNDataLoader(pl.LightningDataModule):
    def __init__(self, root, name , batchsize = 16,num_workers=24):
        super().__init__()
        self.root = root
        self.name = name
        self.batchsize = batchsize
        self.num_workers = num_workers
        self.transform = self.get_transform()
            
    def get_transform(self):
        transform = [
            ToTensor()
        ]
        return Compose(transform)

    def setup(self, stage: Optional[str] = None):
        self.train_data = MaskRCNNDataset(self.root,self.transform,self.name)
        self.val_data = MaskRCNNDataset(self.root,self.transform,self.name)
        print('dataset length :', len(self.train_data))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data,batch_size = self.batchsize,num_workers=self.num_workers)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,batch_size = self.batchsize,num_workers=self.num_workers)