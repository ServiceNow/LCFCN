
from PIL import Image
import numpy as np
import torch
import imageio
import os
import copy
from haven import haven_utils as hu
# from src import proposals
from src import datasets
from skimage.io import imread
from scipy.io import loadmat
import torchvision.transforms.functional as FT
import numpy as np
import torch
from skimage import morphology
from skimage.io import imread
import torchvision.transforms.functional as FT
from skimage.transform import rescale
import torchvision
from torchvision.transforms import transforms
import pylab as plt
from skimage.color import label2rgb
from skimage.segmentation import slic
from haven import haven_utils as hu
from haven import haven_img as hi
# from repos.aranxta_code.extract_cost import CsObject
# from repos.selectivesearch.selectivesearch import selective_search
# from src import region_methods
# import pycocotools.mask as mask_util
from skimage.segmentation import mark_boundaries
import pandas as pd 


class Pascal:
    def __init__(self, split, datadir, exp_dict, sbd=False):     
        self.split = split
        self.exp_dict = exp_dict
        # self.n_classes = 21
        self.n_classes = 20
        self.datadir = datadir
        if split == "train":
            # berkeley addition of images
            if sbd == True:           
                dataset = torchvision.datasets.SBDataset(os.path.join(datadir, 'sbdataset'),
                                                           image_set='train',
                                                           download=False)
            else:
                dataset = torchvision.datasets.VOCSegmentation(datadir,
                                                            year='2012',
                                                            image_set='train',
                                                            download=False)
            
        elif split in ["val", 'test']:
            dataset = torchvision.datasets.VOCSegmentation(datadir,
                                                            image_set='val',
                                                            download=False)
            
        self.point_dict = hu.load_json(os.path.join(datadir, 'VOCdevkit',                 
                                    'pascal2012_trainval_main.json'))
        self.dataset = dataset
        self.transforms = None
                                                                         
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # index = 0
        img_path = self.dataset.images[index]
        name = os.path.split(img_path)[-1].split('.')[0]

        img_pil = Image.open(img_path).convert("RGB")
        W, H = img_pil.size
        points_list = self.point_dict[name]
        points_mask = np.zeros((H, W))
        for p in points_list:
            if p['y'] >= H or p['x'] >= W:
                continue
            points_mask[int(p['y']), int(p['x'])] = p['cls']
        
        images = torchvision.transforms.ToTensor()(np.array(img_pil))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        images = transforms.Normalize(mean=mean, std=std)(images)
        counts = np.zeros(20)
        uni, cts = np.unique(points_mask, return_counts=True)
        if len(uni) > 1:
            counts[uni[1:].astype(int) - 1] = cts[1:]
        return {"images":images, 
                "points":torch.as_tensor(points_mask.squeeze()), 
                "counts":torch.as_tensor(counts), 
                'meta':{"index":index}}



def get_blob_list(mask_dict, points_mask, img_pil, split_inst=False):
    blob_list = []
    mask = preds = mask_dict['preds']
    probs = mask_dict['probs']
    assert probs.shape[1] == preds.shape[0]
    assert probs.shape[2] == preds.shape[1]
    
    imask = np.zeros(mask.shape)
    cmask = np.zeros(mask.shape)
    
    blob_id = 1
    for c in np.unique(mask):
        if c == 0:
            continue
        probs_class = probs[c]
        point_ind = points_mask == c
        inst_mask = morphology.label(mask==c)
        for l in np.unique(inst_mask):
            if l == 0:
                continue
            blob_ind = inst_mask == l
            locs = np.where(blob_ind * point_ind)
            y_list, x_list = locs
            n_points = len(y_list)
            if n_points == 0:
                continue
            if n_points > 1 and split_inst: 
                # split multiple points
                img_points = hi.points_on_image(y_list, x_list, img_pil)
                img_masks = hi.mask_on_image(img_pil, mask)
                img_masks = hi.mask_on_image(img_points.copy(), blob_ind)
                hu.save_image('tmp.jpg', (img_points)*0.5 + hu.f2l(hi.gray2cmap(probs_class))*0.5)
                hu.save_image('tmp.jpg', img_masks)
                
                for yi, xi in zip(y_list, x_list):
                    imask, cmask, blob_list, blob_id = add_mask(yi, xi, points_mask, 
                                                            blob_ind, 
                                                            n_points, blob_list, imask, cmask,
                                                            blob_id)
            else:
                # add for that single point
                yi, xi = y_list[0], x_list[0]
                imask, cmask, blob_list, blob_id = add_mask(yi, xi, points_mask, 
                                                            blob_ind, 
                                                            n_points, blob_list, imask, cmask,
                                                            blob_id)
                

    return blob_list, cmask.astype('uint8'), imask.astype('uint8')

def get_blob_list_v2(mask_dict, points_mask, img_pil):
    blob_list = []
    preds = mask_dict['preds']
    probs = mask_dict['probs']

    masker = Masker(preds)

    for c in np.unique(preds):
        if c == 0:
            continue
        point_ind = points_mask == c
        inst_mask = morphology.label(preds==c)
        for l in np.unique(inst_mask):
            if l == 0:
                continue
            blob_ind = inst_mask == l
            locs = np.where(blob_ind * point_ind)
            y_list, x_list = locs
            n_points = len(y_list)
            if n_points == 0:
                continue
            if n_points > 1:
                # split multiple points
                mask_list = watersplit(probs[l,:,:], (blob_ind * point_ind), img_pil, blob_ind)
                assert(len(mask_list) == n_points)
                for mask_dict in mask_list:
                    yi = mask_dict['yi']
                    xi = mask_dict['xi']

                    masker.add_mask(yi, xi, points_mask, mask_dict['mask'],
                                    n_points)
            else:
                # add for that single point
                yi, xi = y_list[0], x_list[0]
                masker.add_mask(yi, xi, points_mask, blob_ind,
                                    n_points)

    y_list, x_list = np.where(points_mask)
    # 
    masker.save(img_pil, y_list, x_list, preds )
    # mask
    blob_list, cmask, imask =  masker.get_masks()
    masker.save(img_pil, y_list, x_list, imask )
    return blob_list, cmask.astype('uint8'), imask.astype('uint8')

class Masker:
    def __init__(self,mask):
        self.imask = np.zeros(mask.shape, 'uint8')
        self.cmask = np.zeros(mask.shape, 'uint8')
        self.blob_list = []
        self.blob_id = 1

    def add_mask(self, yi, xi, points_mask, blob_ind, n_points):
        ci = points_mask[yi, xi]
        self.blob_list += [{'mask':blob_ind, 'point':{'y':yi, 'xi':xi}, 'cls':ci}]
        self.imask[blob_ind] = self.blob_id
        self.cmask[blob_ind] = ci
        self.blob_id += 1


    def get_masks(self):
        return self.blob_list, self.cmask, self.imask

    def save(self, img_pil, y_list, x_list, mask):
        img_p = hi.points_on_image( y_list, x_list, img_pil)

        img_maskspil = hi.mask_on_image(img_p, mask.astype('uint8'), add_bbox=True)
        hu.save_image('masker.jpg', img_maskspil)

def watersplit(_probs, _points, img_pil, blob_ind):
    import numpy as np
    from skimage.morphology import watershed
    from skimage.segmentation import find_boundaries
    from scipy import ndimage
    from scipy import ndimage as ndi

    points = _points.copy()
    img_pilb = img_pil.convert('L')

    # points[points!=0] = np.arange(1, points.sum()+1)
    points = ndi.label(points)[0]
    points = points.astype(float)

    # probs = ndimage.black_tophat(_probs.copy(), 7)
    probs = ndimage.black_tophat(_probs.clone(), 7)
    # seg =  watershed(probs, points, mask=img_pilb)
    water_mask =  watershed(probs, points, mask=blob_ind)
    mask_list = []
    for p in np.unique(points):
        if p == 0:
            continue 
        y_list, x_list = np.where(points==p)
        mask_list += [{'yi':y_list[0], 'xi':x_list[0], 'mask':water_mask==p}]
    
    # for i, mask_dict in enumerate(mask_list):
    #     img_maskspil = hi.mask_on_image(img_pil, mask_dict['mask'])
    #     img_points = hi.points_on_image([mask_dict['yi']], [mask_dict['xi']], img_maskspil)
    #     hu.save_image('water_mask_%d.jpg' % i, img_points)
    return mask_list