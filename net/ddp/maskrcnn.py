import torch
import torchvision
import pytorch_lightning as pl
import torch.distributed as dist
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class MaskRCNN(pl.LightningModule):
    def __init__(self, lr = 1e-3):
        super().__init__()
        self.lr = lr
        self.model = self.get_model_instance_segmentation(2)

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

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        super().on_train_start()
        self.model = self.model.train()

    def training_step(self, batch, batch_idx):
        x,y = batch

        images = x
        target = y
        
        loss_dict = self.model(batch)

        losses = sum(loss for loss in loss_dict.values())
        '''
        loss_dict_reduced = self.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        '''

        self.log("training_loss", loss_dict, on_step=True, on_epoch=True, sync_dist=True)
        return losses

    def validation_step(self, batch, batch_idx):
        images,targets = batch
        x = list(image for image in images)
        print(targets)
        y = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(x,y)
        self.log("validation_loss", loss_dict, on_step=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x,y = batch
        images = x
        target = y
        loss_dict = self.model(batch)
        self.log("test_loss", loss_dict, on_step=True, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x,y = batch
        return self.model(x)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.lr,
                            momentum=0.9, weight_decay=0.0005)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=3,
                                                        gamma=0.1),
                "monitor": "metric_to_track",
                "frequency": "indicates how often the metric is updated"
            },
        }
    def reduce_dict(self,input_dict, average=True):
        world_size = self.get_world_size()
        if world_size < 2:
            return input_dict
        with torch.no_grad():
            names = []
            values = []
            # sort the keys so that they are consistent across processes
            for k in sorted(input_dict.keys()):
                names.append(k)
                values.append(input_dict[k])
            values = torch.stack(values, dim=0)
            dist.all_reduce(values)
            if average:
                values /= world_size
            reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict

    def get_world_size(self):
        if not self.is_dist_avail_and_initialized():
            return 1
        return dist.get_world_size()

    def is_dist_avail_and_initialized(self):
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True