from torch.utils import data

import numpy as np
import torch
import os
from skimage.io import imread
# import torchvision.transform.functional as FT

from skimage.transform import rescale
import utils as ut 

camera_list = ["BAILa","DAMOa","GEORa","HALFb","HALFc",
                "LOCKb","MAIVb","MAIVc","NEKOa","NEKOb",
                "NEKOc","PETEc","PETEd","PETEe","PETEf",
                "SPIGa"]




def get_corrupted_images(path):
    corrupted = []
    n_total = 0
    for camera in camera_list:
        pointsDict = ut.load_pkl(path + "CompleteAnnotations_2016-07-11/%s.pkl" % camera)
        n_local = 0
        for img_name in pointsDict:
            if pointsDict[img_name] is None:
                corrupted += [img_name]
                n_local += 1
        print("%s:%d" % (camera, n_local))
        n_total += len(pointsDict)
    print("%d/%d corrupted" % (len(corrupted), n_total))
    return corrupted


def save_pointsDict(path, imgNames):
    allPointDict = {}

    for camera in camera_list:
        print(camera)
        pointsDict = ut.load_pkl(path + 
                                "CompleteAnnotations_2016-07-11/%s.pkl" % camera)
        for img_name in pointsDict:
            if img_name in imgNames:
                allPointDict[img_name] = pointsDict[img_name]
    return allPointDict




class PenguinsMixed(data.Dataset):
    def __init__(self, root="", split=None, 
                 transform_function=None, ratio=0.6):
        
        self.split = split
        self.path = "/mnt/datasets/public/issam/Penguins/"
        
        imgNames = ut.load_pkl(self.path + 
                               "/Splits_2016_07_11/%s_mixed.pkl" % split)
             
        self.imgNames = imgNames
        path_points = (self.path + "/Splits_2016_07_11/points_%s_mixed.pkl" % split)

        # Load point dict for each camera
        if not os.path.exists(path_points):
            allPointDict = save_pointsDict(self.path, self.imgNames)
            ut.save_pkl(path_points, allPointDict)

        self.pointsDict = ut.load_pkl(path_points)
        

        assert len(self.pointsDict) == len(self)

        self.split = split
        self.n_classes = 2
        self.transform_function = transform_function

        roi_name = self.path + "CompleteAnnotations_2016-07-11/roi.pkl"
        self.roiDict = ut.load_pkl(roi_name)
        

    def __len__(self):
        return len(self.imgNames)

    def __getitem__(self, index):
        img_name = self.imgNames[index]
        camera = img_name[:5]
        img_org = imread(self.path + "%s/%s.JPG"  % (camera, img_name))
        img = rescale(img_org, self.ratio, mode="constant", preserve_range=True).astype(np.uint8)

        h,w = img.shape[:2]

        long_name = img_name.split("_")[0]
        cols = self.roiDict[long_name]["cols"] * self.ratio
        rows = self.roiDict[long_name]["rows"] * self.ratio
        roi = ut.poly2mask(rows.astype(int), cols.astype(int), shape=(h,w))
        roi = roi[:,:,np.newaxis]


        # GET ANN LISTS
        assert self.pointsDict[img_name] is not None
        pointsList = ut.get_longest_list(self.pointsDict[img_name])
       
        points = np.zeros((h,w,1), np.uint8)
        for p in pointsList:
            x, y = p
            ys = int(y * self.ratio)
            xs = int(x * self.ratio)

            if ys >= h or xs >= w or xs < 0 or ys < 0:
                continue
            else:
                points[ys, xs]=1

        # LOAD IMG AND POINT
        org = torch.FloatTensor(ut.shrink2roi(img, roi))
        img = ut.shrink2roi(img * roi, roi)
        points = ut.shrink2roi(points * roi, roi)

        collection = list(map(FT.to_pil_image, [img, points]))

        counts = int(points.sum())
        
        if self.transform_function is not None:
            _img, _points = self.transform_function(collection)

        if np.all(_points == -1):
            pass
        else:
            assert int(_points.sum()) == counts

        medianPointsList = ut.get_median_list(self.pointsDict[img_name])
        median_count = points2count(medianPointsList, self.ratio, h, w, roi)
        assert  points2count(pointsList, self.ratio, h, w, roi) == counts
        
        counts = torch.LongTensor([counts])

        return {"images":_img, "points":_points, 
                "counts":counts, "index":index,"org":org,
                "counts_median":torch.LongTensor([median_count])}


def points2count(pointsList, ratio, h, w, roi):
    points = np.zeros((h,w,1), np.uint8)
    for p in pointsList:
        x, y = p
        ys = int(y * ratio)
        xs = int(x * ratio)

        if ys >= h or xs >= w or xs < 0 or ys < 0:
            continue
        else:
            points[ys, xs]=1

    points = ut.shrink2roi(points * roi, roi)
    return int(points.sum())
