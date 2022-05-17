import warnings
warnings.filterwarnings(action='ignore') 
import torch
import os
import argparse
import numpy as np
from pytorch_lightning import Trainer
from net.ddp.maskrcnn import MaskRCNN
from util.ddp.dataloader.maskrcnn_dataloader import MaskRCNNDataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin


def get_args():
	parser = argparse.ArgumentParser(description='Arguments for the testing purpose.')	
	parser.add_argument('--batch_size', type=int, required=False, default=32)
	parser.add_argument('--num_gpus', type=int, required=False, default=3)
	parser.add_argument('--num_epochs', type=int, required=False, default=2)
	parser.add_argument('--learning_rate', type=int, required=False, default=1e-3)
	parser.add_argument('--name', type=str, required=False, default='jururu')
	parser.add_argument('--model_save', type=bool, required=False, default=True)
	parser.add_argument('--data_path', type=str, required=False, default='/home/ubuntu/jini1114/augmen/')
	parser.add_argument('--model_path', type=str, required=False, default='/home/ubuntu/jini1114/ddp_test/model')
	args = parser.parse_args()

	return args

if __name__ == "__main__":
	#get arg
	args = get_args()
	print(args)

	#set logger
	wandb_logger = WandbLogger(project="multi-gpu-maskrcnn")
	wandb_logger.config = args

	#dataloader loading
	dl = MaskRCNNDataLoader(root = args.data_path, name = args.name, batchsize= args.batch_size)
	dl.setup()

	#setup model
	ddpmaskrcnn = MaskRCNN(mode = 'train', lr = args.learning_rate)
	
	#setup trainer
	trainer = Trainer(
		max_epochs=args.num_epochs, 
		gpus=args.num_gpus, 
		accelerator="ddp",
		plugins=DDPPlugin(find_unused_parameters=False),
		logger = wandb_logger
	)

	#training
	trainer.fit(ddpmaskrcnn, datamodule=dl)

	if args.model_save :
		print('training model save')
		trainer.save_checkpoint(os.path.join(args.model_path,'./ddp_%s_maskrcnn.pth'%(args.name)))

		#model to onnx
		X = torch.tensor(np.zeros([128,3,680,720])).to(torch.float)
		torch.onnx.export(ddpmaskrcnn,                     # model being run
						X,              # model input (or a tuple for multiple inputs)
						os.path.join(args.model_path,"ddp_%s_maskrcnn.onnx"%(args.name)), # where to save the model (can be a file or file-like object)
						export_params=True,        # store the trained parameter weights inside the model file
						opset_version=10,          # the ONNX version to export the model to
						do_constant_folding=True,  # whether to execute constant folding for optimization
						input_names = ['input'],   # the model's input names
						output_names = ['output'], # the model's output names
						dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
										'output' : {0 : 'batch_size'}}) 
	else :
		print('training model doesnt save')