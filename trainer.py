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

#to onnx
X = torch.tensor(np.zeros([128,3,680,720])).to(torch.float)

torch.onnx.export(ddpautoencoder,                     # model being run
                  ##since model is in the cuda mode, input also need to be
                  X,              # model input (or a tuple for multiple inputs)
                  "ddpautoencoder.onnx", # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})