import os
import torch
import cv2
import numpy as np

class MaskRCNNDataset(object):
    def __init__(self, root, transforms,name, num=0):
        self.root = root
        self.transforms = transforms
        self.name = name
        self.files = os.listdir(os.path.join(root,'Masks'))

    def __getitem__(self, idx):
        
        thrshold = 150
        img = cv2.imread(os.path.join(self.root,'Images',self.files[idx][:-3]+'jpg'))
        img = img.astype(np.float32)
        img = cv2.resize(img, dsize=(580, 620))
        img = img/255

        mask = cv2.imread(os.path.join(self.root,'Masks',self.files[idx]))
        mask = mask[:,:,0]
        mask = cv2.resize(mask, dsize=(580, 620))
        mask[mask <thrshold] = 0
        mask[mask >=thrshold] = 1

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmin == xmax:
                xmax = xmax+1
            if ymin == ymax :
                ymax = ymax+1
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = torch.tensor(100)
        target["iscrowd"] = iscrowd

        if self.transforms is not None:

            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.files)