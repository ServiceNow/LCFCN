from torch.utils import data
import numpy as np
import torch
import os
from skimage.io import imread
from scipy.io import loadmat
import utils as ut
import torchvision.transforms.functional as FT



class Trancos(data.Dataset):
    def __init__(self, root="",split=None, 
                 transform_function=None):
        self.split = split
        
        self.n_classes = 2
        self.transform_function = transform_function
        
        ############################
        self.path_base = "datasets/TRANCOS_v3/"
        # self.path_base = "/mnt/datasets/public/issam/Trancos/"

        if split == "train":
            fname = self.path_base + "/image_sets/training.txt"

        elif split == "val":
            fname = self.path_base + "/image_sets/validation.txt"

        elif split == "test":
            fname = self.path_base + "/image_sets/test.txt"

        self.img_names = [name.replace(".jpg\n","") for name in ut.read_text(fname)]
        self.path = self.path_base + "/images/"
        assert os.path.exists(self.path + self.img_names[0] + ".jpg")
        

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        name = self.img_names[index]

        # LOAD IMG, POINT, and ROI
        image = imread(self.path + name + ".jpg")
        points = imread(self.path + name + "dots.png")[:,:,:1].clip(0,1)
        roi = loadmat(self.path + name + "mask.mat")["BW"][:,:,np.newaxis]
        
             

        # LOAD IMG AND POINT
        image = image * roi
        image = ut.shrink2roi(image, roi)
        points = ut.shrink2roi(points, roi).astype("uint8")

        counts = torch.LongTensor(np.array([int(points.sum())]))   
        
        collection = list(map(FT.to_pil_image, [image, points]))
        
        if self.transform_function is not None:
            image, points = self.transform_function(collection)
            
        if np.all(points == -1):
            pass
        else:
            assert int(points.sum()) == counts[0]

        return {"images":image, "points":points, 
                "counts":counts, "index":index,
                "image_path":self.path + name + ".jpg"}
