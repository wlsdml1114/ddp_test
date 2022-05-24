import torch
import torchvision
import pytorch_lightning as pl
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class MaskRCNN(pl.LightningModule):
    def __init__(self, mode = 'train',lr = 1e-3):
        super().__init__()
        self.lr = lr
        model = self.get_model_instance_segmentation(2)
        if mode == 'train':
            self.model = model.train()
        else :
            self.model = model.eval()
        self.keys = ['boxes', 'labels','masks','image_id','area','iscrowd']

    def get_model_instance_segmentation(self, num_classes):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)

        return model

    def forward(self, batch):
        x,y = batch
        return self.model(x,y)

    def training_step(self, batch, batch_idx):
        self.model.train()

        x,y = self.batch_reconstruction(batch)

        loss_dict = self.model(x,y)

        losses = sum(loss for loss in loss_dict.values())

        loss_dict = {t:loss_dict[t].detach() for t in loss_dict}
        
        self.log("training_loss", loss_dict, on_step=True, on_epoch=True, sync_dist=True)
        return losses

    def validation_step(self, batch, batch_idx):
        self.model.train()
  
        x,y = self.batch_reconstruction(batch)

        loss_dict = self.model(x,y)

        losses = sum(loss for loss in loss_dict.values())

        loss_dict = {t:loss_dict[t].detach() for t in loss_dict}

        self.log("validation_loss", loss_dict, on_step=True, on_epoch=True, sync_dist=True)
        return losses
    def test_step(self, batch, batch_idx):
        x,y = self.batch_reconstruction(batch)
        loss_dict = self.model(x,y)
        self.log("test_loss", loss_dict, on_step=True, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x,y = batch
        return self.model(x)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.lr,
                            momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=3,
                                                        gamma=0.1)
        return [optimizer],[scheduler]

    def batch_reconstruction(self, batch):
        images,targets = batch
        x = list(image for image in images)
        y = []
        for idx in range(len(x)):
            dict = {}
            for key in self.keys:
                dict[key] = targets[key][idx]
            y.append(dict)
        return x, y