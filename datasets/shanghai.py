from torch.utils import data
import glob
import numpy as np
import torch
import os
from skimage.io import imread

import torchvision.transforms.functional as FT
import utils as ut


class ShanghaiB(data.Dataset):
    def __init__(self, split=None, 
                 transform_function=None):
        self.transform_function = transform_function
        self.split = split
        self.n_classes = 2
        self.root = "/mnt/datasets/public/issam/ShanghaiTech/part_B/"

        if split == "train":
            self.path = self.root + "/train_data/"
            self.img_names = [os.path.basename(x) for x in 
                                glob.glob(self.path+"/images/*")][:-50]

        elif split == "val":
            self.path = self.root + "/train_data/"
            self.img_names = [os.path.basename(x) for x in 
                                glob.glob(self.path+"/images/*")][-50:]

        elif split == "test":
            self.path = self.root + "/test_data/"
            self.img_names = [os.path.basename(x) for x in 
                                glob.glob(self.path+"/images/*")]    

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):        
        name = self.img_names[index]
      
        # LOAD IMG, POINT, and ROI
        image = imread(self.path + "/images/" + name)
        if image.ndim == 2:
            image = image[:,:,None].repeat(3,2)
        pointList = ut.loadmat(self.path + "/ground-truth/GT_" + name.replace(".jpg", "") +".mat")
        pointList = pointList["image_info"][0][0][0][0][0] 
        
        points = np.zeros(image.shape[:2], "uint8")[:,:,None]
        for x, y in pointList:
            points[int(y), int(x)] = 1

        counts = torch.LongTensor(np.array([pointList.shape[0]]))

        collection = list(map(FT.to_pil_image, [image, points]))
        if self.transform_function is not None:
            image, points = self.transform_function(collection)


        return {"images":image, "points":points, 
                "counts":counts, "index":index,
                "image_path":self.path + "/images/" + name}