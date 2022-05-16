import warnings
warnings.filterwarnings(action='ignore') 
import torch
import os
import argparse
import time
import numpy as np
from tqdm import tqdm
from net.ddp.autoencoder import AutoEncoder


def get_args():
	parser = argparse.ArgumentParser(description='Arguments for the testing purpose.')	
	parser.add_argument('--data_path', type=str, required=False, default='/home/ubuntu/jini1114/dataset/')
	parser.add_argument('--name', type=str, required=False, default='viichan')
	parser.add_argument('--model_path', type=str, required=False, default='/home/ubuntu/jini1114/ddp_test/model')
	args = parser.parse_args()

	return args

#get arg
args = get_args()

#set root
data_path = os.path.join(args.data_path,args.name)

#model loading
model = AutoEncoder()
model.load_from_checkpoint(os.path.join(args.model_path,'ddpautoencoder.pth'))
model.eval()
model.cuda()

#onnx input setting
X = torch.tensor(np.zeros([128,3,680,720]).astype(np.float32)).cuda()

# time count
start = time.time()

#run
for i in tqdm(range(100)):
	results = model(X)

# time count
print('time : ',time.time() - start)