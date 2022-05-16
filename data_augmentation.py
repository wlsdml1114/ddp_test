import os
import cv2
import argparse
import numpy as np
import torchvision.transforms.functional as TF
from tqdm import tqdm
from PIL import Image


def get_args():
	parser = argparse.ArgumentParser(description='Arguments for the testing purpose.')	
	parser.add_argument('--ori_dir', type=str, required=True)
	parser.add_argument('--aug_dir', type=str, required=True)
	args = parser.parse_args()

	return args

#get arg
args = get_args()

ori_dir = args.ori_dir
aug_dir = args.aug_dir

folders = os.listdir(ori_dir)

for folder in folders:

    print(folder,'data augmentation start')

    for num in os.listdir(os.path.join(ori_dir,folder)):
        if 'DS_Store' in num:
            continue
        files = os.listdir(os.path.join(ori_dir,folder,num,"Masks"))
        if not(os.path.exists(os.path.join(aug_dir,folder,"Images"))):
            os.system('mkdir -p '+os.path.join(aug_dir,folder,"Images"))

        if not(os.path.exists(os.path.join(aug_dir,folder,"Masks"))):
            os.system('mkdir -p '+os.path.join(aug_dir,folder,"Masks"))

        for idx in tqdm(range(len(files))):
            try :
                img = cv2.imread(os.path.join(ori_dir,folder,num,"Images",files[idx][:-3]+'jpg'))      
                
                edges = cv2.Canny(image=img, threshold1=50, threshold2=200) # Canny Edge Detection
                img2 = cv2.merge((edges,edges,edges))

                kernel = np.ones((2,2),np.uint8)
                dilation = cv2.dilate(img2,kernel,iterations = 1)
                kernel = np.ones((4,4),np.uint8)
                dilation2 = cv2.dilate(img2,kernel,iterations = 1)

                img = img.astype(np.int16)
                img = img +dilation2
                img = img - dilation
                img[img <0] = 0
                img[img > 255] = 255
                img = img.astype(np.uint8)
            
                pil_image = Image.fromarray(img)
            except :
                print(folder,files[idx],"Exception occur")
                continue
            flip_img = TF.hflip(pil_image)
            bright_img=TF.adjust_brightness(pil_image,0.7)
            cont_img=TF.adjust_contrast(pil_image,0.8)
            crop_img = img[100:,100:]
            cv2.imwrite(os.path.join(aug_dir,folder,"Images",'ori_%s_%sjpg'%(num,files[idx][:-3])),img)
            cv2.imwrite(os.path.join(aug_dir,folder,"Images",'flip_%s_%sjpg'%(num,files[idx][:-3])),np.array(flip_img))
            cv2.imwrite(os.path.join(aug_dir,folder,"Images",'bri_%s_%sjpg'%(num,files[idx][:-3])),np.array(bright_img))
            cv2.imwrite(os.path.join(aug_dir,folder,"Images",'cont_%s_%sjpg'%(num,files[idx][:-3])),np.array(cont_img))
            cv2.imwrite(os.path.join(aug_dir,folder,"Images",'crop_%s_%sjpg'%(num,files[idx][:-3])),crop_img)



            img = cv2.imread(os.path.join(ori_dir,folder,num,"Masks",files[idx]))
            pil_image = Image.fromarray(img)
            flip_img = TF.hflip(pil_image)
            bright_img=TF.adjust_brightness(pil_image,0.7)
            cont_img=TF.adjust_contrast(pil_image,0.8)
            crop_img = img[100:,100:]
            cv2.imwrite(os.path.join(aug_dir,folder,"Masks",'ori_%s_%spng'%(num,files[idx][:-3])),img)
            cv2.imwrite(os.path.join(aug_dir,folder,"Masks",'flip_%s_%spng'%(num,files[idx][:-3])),np.array(flip_img))
            cv2.imwrite(os.path.join(aug_dir,folder,"Masks",'bri_%s_%spng'%(num,files[idx][:-3])),np.array(bright_img))
            cv2.imwrite(os.path.join(aug_dir,folder,"Masks",'cont_%s_%spng'%(num,files[idx][:-3])),np.array(cont_img))
            cv2.imwrite(os.path.join(aug_dir,folder,"Masks",'crop_%s_%spng'%(num,files[idx][:-3])),crop_img)