from util.ddp.dataloader.autoencoder_dataloader import AutoEncoderDataLoader
import os


data_path = os.path.join('/home/ubuntu/jini1114/dataset/','viichan')

dl = AutoEncoderDataLoader(root = data_path, batchsize=128)
dl.setup()

for batch in dl.train_dataloader():
    print(len(batch))

    # dataloader
    # ddpPlugin for https://github.com/PyTorchLightning/pytorch-lightning/discussions/6761
    # iter https://discuss.pytorch.org/t/iterating-through-a-dataloader-object/25437
    # has issue https://github.com/PyTorchLightning/pytorch-lightning/issues/3175
    