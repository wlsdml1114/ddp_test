import os
from util.ddp.dataloader.autoencoder_dataloader import AutoEncoderDataLoader


data_path = os.path.join('/home/ubuntu/jini1114/dataset/','viichan')

dl = AutoEncoderDataLoader(root = data_path, batchsize=128)
dl.setup()

for batch in dl.train_dataloader():
    print(len(batch))
    