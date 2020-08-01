from torch.utils import data
import glob
import numpy as np
import torch
import os
from skimage.io import imread
from haven import haven_utils as hu
import torchvision.transforms.functional as FT
from . import transformers


class Shanghai(data.Dataset):
    def __init__(self, split, datadir, exp_dict):
        self.split = split
        self.n_classes = 1
        self.exp_dict = exp_dict

        if split == "train":
            self.path = os.path.join(datadir, "train_data")
            self.img_names = [os.path.basename(x) for x in 
                                glob.glob(os.path.join(self.path,"images", "*"))][:-50]

        elif split == "val":
            self.path = os.path.join(datadir, "train_data")
            self.img_names = [os.path.basename(x) for x in 
                                glob.glob(os.path.join(self.path,"images", "*"))][-50:]

        elif split == "test":
            self.path = os.path.join(datadir, "test_data")
            self.img_names = [os.path.basename(x) for x in 
                                glob.glob(os.path.join(self.path,"images", "*"))]    

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):        
        name = self.img_names[index]
      
        # LOAD IMG, POINT, and ROI
        image = imread(os.path.join(self.path, "images", name))
        if image.ndim == 2:
            image = image[:,:,None].repeat(3,2)
        pointList = hu.load_mat(os.path.join(self.path, 
                        "ground-truth", 
          "GT_" + name.replace(".jpg", "") +".mat"))
        pointList = pointList["image_info"][0][0][0][0][0] 
        
        points = np.zeros(image.shape[:2], "uint8")[:,:,None]
        H, W = image.shape[:2]
        for x, y in pointList:
            points[min(int(y), H-1), min(int(x), W-1)] = 1

        counts = torch.LongTensor(np.array([pointList.shape[0]]))

        collection = list(map(FT.to_pil_image, [image, points]))
        image, points = transformers.apply_transform(self.split, image, points, 
                   transform_name=self.exp_dict['dataset']['transform'])
            
        return {"images":image, 
                "points":points.squeeze(), 
                "counts":counts, 
                'meta':{"index":index}}