import warnings
warnings.filterwarnings(action='ignore') 
from pytorch_lightning import Trainer
from net.ddp.autoencoder import AutoEncoder
from util.ddp.dataloader.autoencoder_dataloader import AutoEncoderDataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
import torch
import os
import numpy as np
import argparse


def get_args():
	parser = argparse.ArgumentParser(description='Arguments for the testing purpose.')	
	parser.add_argument('--batch_size', type=int, required=False, default=128)
	parser.add_argument('--num_gpus', type=int, required=False, default=3)
	parser.add_argument('--num_epochs', type=int, required=False, default=10)
	parser.add_argument('--learning_rate', type=int, required=False, default=1e-3)
	parser.add_argument('--name', type=str, required=False, default='viichan')
	parser.add_argument('--data_path', type=str, required=False, default='/home/ubuntu/jini1114/dataset/')
	parser.add_argument('--model_path', type=str, required=False, default='/home/ubuntu/jini1114/ddp_test/model')
	args = parser.parse_args()

	return args

if __name__ == "__main__":
	#get arg
	args = get_args()
	print(args)

	#set root
	data_path = os.path.join(args.data_path,args.name)

	#set logger
	wandb_logger = WandbLogger(project="multi-GPU", entity="engui")
	wandb_logger.config = args

	#dataloader loading
	dl = AutoEncoderDataLoader(root = data_path, batchsize= args.batch_size)
	dl.setup()

	#setup model
	ddpautoencoder = AutoEncoder(root = data_path, lr = args.learning_rate)
	
	#setup trainer
	trainer = Trainer(
		max_epochs=args.num_epochs, 
		gpus=args.num_gpus, 
		accelerator="ddp",
		plugins=DDPPlugin(find_unused_parameters=False),
		logger = wandb_logger
	)

	#training
	trainer.fit(ddpautoencoder, datamodule=dl)
	trainer.save_checkpoint(os.path.join(args.model_path,'./ddpautoencoder.pth'))

	#model to onnx
	X = torch.tensor(np.zeros([128,3,680,720])).to(torch.float)
	torch.onnx.export(ddpautoencoder,                     # model being run
					X,              # model input (or a tuple for multiple inputs)
					os.path.join(args.model_path,"ddpautoencoder.onnx"), # where to save the model (can be a file or file-like object)
					export_params=True,        # store the trained parameter weights inside the model file
					opset_version=10,          # the ONNX version to export the model to
					do_constant_folding=True,  # whether to execute constant folding for optimization
					input_names = ['input'],   # the model's input names
					output_names = ['output'], # the model's output names
					dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
									'output' : {0 : 'batch_size'}})