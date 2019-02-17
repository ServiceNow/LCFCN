from torch.utils import data
import numpy as np
import torch
import os
from skimage.io import imread

import tqdm
import torchvision.transforut.functional as FT

from skimage import draw
from skimage.transform import rescale
from skimage import morphology as morph
from scipy import ndimage
import utils as ut

camera_list = ["BAILa","DAMOa","GEORa","HALFb","HALFc",
                "LOCKb","MAIVb","MAIVc","NEKOa","NEKOb",
                "NEKOc","PETEc","PETEd","PETEe","PETEf",
                "SPIGa"]


def get_count(annList, penguinSize):
    h, w = penguinSize.shape[:2]
    if len(annList) == 0:
        return 0

    annots = np.zeros((len(annList), h, w))

    for i, pList in enumerate(annList):
        if pList is None:
            continue
        for p in pList:
            if not isinstance(p, list):
                continue

            x, y = p
            if y < h and x < w and x>=0 and y>=0:
                annots[i, y, x] = 1

    penguinSize[penguinSize<20] = 20;
    distTf = ndimage.distance_transform_edt(1.-annots.sum(0).clip(0, 1))
    regions = distTf<(0.5*penguinSize);
    blobs = morph.label(regions)
    count_max = 0
    count_median = 0

    annots_ravel = annots.reshape((len(annList), -1))
    for i in np.unique(blobs):
        if i == 0:
            continue

        n_points = (annots_ravel * (blobs == i).ravel()).sum(1)
        count_max += np.max(n_points)
        count_median += np.median(n_points)

    return {"max":count_max, "median":count_median}


def filterMe(L, corrupted):
    return [ut.extract_fname(s).replace(".JPG","") for s in L
            if ut.extract_fname(s).replace(".JPG","") not in corrupted]

def get_splits(path, version, corrupted):
    corrupted = get_corrupted_images(path)
    splits = ut.load_json(path + "/Splits_2016_07_11/%s.json" % version)

    tr = filterMe(splits["imdb"]["train"], corrupted)
    vl = filterMe(splits["imdb"]["val"], corrupted)
    te = filterMe(splits["imdb"]["test"], corrupted)

    ut.save_pkl(path + "/Splits_2016_07_11/train_%s.pkl" % version, 
                tr)

    ut.save_pkl(path + "/Splits_2016_07_11/val_%s.pkl" % version,
                 vl)

    ut.save_pkl(path + "/Splits_2016_07_11/test_%s.pkl" % version,
                 te)

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

def savePointsDict(path):
    for camera in camera_list:
        ann = ut.load_json(path + "CompleteAnnotations_2016-07-11/%s.json"  % camera)
        pointsDict = {dots["imName"]:dots["xy"] for dots in ann["dots"]}
        ut.save_pkl(path + "CompleteAnnotations_2016-07-11/%s.pkl"  % camera,
                    pointsDict)

def save_n_penguins(imgNames):
    pass


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

def poly2mask(rows, cols, shape):
    assert len(rows) == len(cols)
    fill_row_coords, fill_col_coords = draw.polygon(rows, cols, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True

    return mask

class Penguins_Base(data.Dataset):
    def get_regioncounts(self, n_images=None):
        from multiprocessing import Pool
        pool = Pool(processes=100)

        countDict = {}
        if n_images is None:
            imgNames = self.imgNames
        else:
            imgNames = self.imgNames[:n_images]

        for img_name in tqdm(imgNames):
            camera = img_name[:5]
            long_name = img_name.split("_")[0]
            penguinSize =imread(self.path + "%s/%s_pengSize.png" % (camera, long_name))

            annList = self.pointsDict[img_name]
            count = pool.apply_async(get_count, args=[annList,penguinSize])
            countDict[img_name] = count

        countDict = {k:countDict[k].get() for k in countDict}
        pool.close()

        return countDict

    def __init__(self, root="", split=None, 
                 transform_function=None, sigma = 8.0,
                  version=None, ratio=0.3):
        
        self.split = split
        self.sigma = sigma
        
        self.name = "Penguins"
        self.path = "Penguins/"

        
        imgNames = ut.load_pkl(self.path + 
                    "/Splits_2016_07_11/%s_%s.pkl" % (split, version))
             
        self.imgNames = imgNames
        path_points = (self.path + 
                             "/Splits_2016_07_11/points_%s_%s.pkl" % 
                             (split,version))

        if not os.path.exists(path_points):
            allPointDict = save_pointsDict(self.path, self.imgNames)
            ut.save_pkl(path_points, allPointDict)

        self.pointsDict = ut.load_pkl(path_points)

        assert len(self.pointsDict) == len(self)

        self.split = split
        self.ratio = ratio
        self.n_classes = 2
        self.n_channels = 3
        self.transform_function = transform_function()
        self.version=version
        
        roi_name = self.path + "CompleteAnnotations_2016-07-11/roi.pkl"

        self.roiDict = ut.load_pkl(roi_name)
        self.longest = True

    def __len__(self):
        return len(self.imgNames)


    def __getitem__(self, index):
        img_name = self.imgNames[index]
        camera = img_name[:5]
        #camera, img_name = self.imgNames[index].split("/")
        img_org = imread(self.path + "%s/%s.JPG"  % (camera, img_name))
        img = rescale(img_org, self.ratio, mode="constant", preserve_range=True).astype(np.uint8)

        
        h,w = img.shape[:2]

        long_name = img_name.split("_")[0]
        cols = self.roiDict[long_name]["cols"] * self.ratio
        rows = self.roiDict[long_name]["rows"] * self.ratio
        roi = poly2mask(rows.astype(int), cols.astype(int), shape=(h,w))
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
                "median_counts":torch.LongTensor([median_count])}


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

class Penguins_MixedFull(Penguins_Base):
    def __init__(self,
                 root,
                 split=None,
                 transform_function=None,
                 sigma=8.0):

        super(Penguins_MixedFull, self).__init__(root,
                 split=split,
                 transform_function=transform_function,
                 sigma=sigma,
                 version="imdb",
                 ratio=0.6)

class Penguins_SepFull(Penguins_Base):
    def __init__(self,
                 root,
                 split=None,
                 transform_function=None,
                 sigma=8.0):

        super(Penguins_SepFull, self).__init__(root,
                 split=split,
                 transform_function=transform_function,
                 sigma=sigma,
                 version="imdb_sepsites",
                 ratio=0.6)


class Penguins_Mixed(Penguins_Base):
    def __init__(self,
                 root,
                 split=None,
                 transform_function=None,
                 sigma=8.0):

        super(Penguins_Mixed, self).__init__(root,
                 split=split,
                 transform_function=transform_function,
                 sigma=sigma,
                 version="imdb")

class Penguins_Sep(Penguins_Base):
    def __init__(self,
                 root,
                 split=None,
                 transform_function=None,
                 sigma=8.0):

        super(Penguins_Sep, self).__init__(root,
                 split=split,
                 transform_function=transform_function,
                 sigma=sigma,
                 version="imdb_sepsites")


class Penguins_MixedSize5(Penguins_Base):
    def __init__(self,
                 root,
                 split=None,
                 transform_function=None,
                 sigma=8.0,ratio=0.5):

        super(Penguins_MixedSize5, self).__init__(root,
                 split=split,
                 transform_function=transform_function,
                 sigma=sigma,
                 version="imdb",
                 ratio=0.5)

class Penguins_SepSize5(Penguins_Base):
    def __init__(self,
                 root,
                 split=None,
                 transform_function=None,
                 sigma=8.0,ratio=0.5):

        super(Penguins_SepSize5, self).__init__(root,
                 split=split,
                 transform_function=transform_function,
                 sigma=sigma,
                 version="imdb_sepsites",
                 ratio=0.5)


class Penguins_MixedMini(Penguins_Base):
    def __init__(self,
                 root,
                 split=None,
                 transform_function=None,
                 sigma=8.0):

        super(Penguins_MixedMini, self).__init__(root,
                 split=split,
                 transform_function=transform_function,
                 sigma=sigma,
                 version="imdb",
                 ratio=0.3)

        if self.split == "train":
            np.random.seed(100)
            self.imgNames = np.random.choice(self.imgNames, 5000, replace=False)



class Penguins_SepMini(Penguins_Base):
    def __init__(self,
                 root,
                 split=None,
                 transform_function=None,
                 sigma=8.0):

        super(Penguins_SepMini, self).__init__(root,
                 split=split,
                 transform_function=transform_function,
                 sigma=sigma,
                 version="imdb_sepsites",
                 ratio=0.3)

        if self.split == "train":
            np.random.seed(100)
            self.imgNames = np.random.choice(self.imgNames, 5000, replace=False)

