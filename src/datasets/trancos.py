from torch.utils import data
import numpy as np
import torch
import os
from skimage.io import imread
from scipy.io import loadmat
import torchvision.transforms.functional as FT
from haven import haven_utils as hu
from . import transformers



class Trancos(data.Dataset):
    def __init__(self, split, datadir, exp_dict):
        self.split = split
        self.exp_dict = exp_dict
        
        self.n_classes = 1
        
        if split == "train":
            fname = os.path.join(datadir, 'image_sets', 'training.txt')

        elif split == "val":
            fname = os.path.join(datadir, 'image_sets', 'validation.txt')

        elif split == "test":
            fname = os.path.join(datadir, 'image_sets', 'test.txt')

        self.img_names = [name.replace(".jpg\n","") for name in hu.read_text(fname)]
        self.path = os.path.join(datadir, 'images')
        

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        name = self.img_names[index]

        # LOAD IMG, POINT, and ROI
        image = imread(os.path.join(self.path, name + ".jpg"))
        points = imread(os.path.join(self.path, name + "dots.png"))[:,:,:1].clip(0,1)
        roi = loadmat(os.path.join(self.path, name + "mask.mat"))["BW"][:,:,np.newaxis]
        
        # LOAD IMG AND POINT
        image = image * roi
        image = hu.shrink2roi(image, roi)
        points = hu.shrink2roi(points, roi).astype("uint8")

        counts = torch.LongTensor(np.array([int(points.sum())]))   
        
        collection = list(map(FT.to_pil_image, [image, points]))
        image, points = transformers.apply_transform(self.split, image, points, 
                   transform_name=self.exp_dict['dataset']['transform'])
            
        return {"images":image, 
                "points":points.squeeze(), 
                "counts":counts, 
                'meta':{"index":index}}
