import warnings
warnings.filterwarnings(action='ignore') 
from pytorch_lightning import Trainer
from net.ddpautoencoder import DDPAutoEncoder
from pytorch_lightning.loggers import WandbLogger
import torch
import os
import numpy as np

#set constant
batch_size = 128
num_gpus = 3
num_epochs = 1

#set root
name = 'viichan'
data_path = '/home/ubuntu/jini1114/dataset/%s/1/'%(name)
model_path = '/home/ubuntu/jini1114/ddp_test/model/'

#set logger
wandb_logger = WandbLogger(project="multi-GPU", entity="engui")
wandb_logger.config = {
  "epochs": num_epochs,
  "batch_size": batch_size,
  "num_gpus" : num_gpus

}

#training
ddpautoencoder = DDPAutoEncoder(data_path,batch_size)
trainer = Trainer(max_epochs=num_epochs, gpus=num_gpus, accelerator="ddp",logger = wandb_logger)
trainer.fit(ddpautoencoder)
trainer.save_checkpoint(os.path.join(model_path,'./ddpautoencoder.pth'))
