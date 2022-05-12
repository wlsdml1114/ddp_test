import warnings
warnings.filterwarnings(action='ignore') 
import torch
import os
import numpy as np
import argparse
import onnxruntime


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
data_path = os.path.join(args.data_path,args.name,'1')

#model loading
session = onnxruntime.InferenceSession(os.path.join(args.model_path,'ddpautoencoder.onnx'),providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
session.get_modelmeta()

#onnx input setting
X = np.zeros([2,3,680,720]).astype(np.float32)

#run
results = session.run([], {"input": X})

print(results)