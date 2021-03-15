import collections
import torch
import random
import numpy as np
import json
from torch.utils import data
import scipy.misc
import scipy.io as io
from skimage import draw
import losses
from PIL import ImageOps
from bs4 import BeautifulSoup
import pickle 
from skimage.segmentation import mark_boundaries


# Misc Utils
def shrink2roi(img, roi):
    ind = np.where(roi != 0)

    y_min = min(ind[0])
    y_max = max(ind[0])

    x_min = min(ind[1])
    x_max = max(ind[1])

    return img[y_min:y_max, x_min:x_max]

def t2n(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()

    return x

def read_text(fname):
    # READS LINES
    with open(fname, "r") as f:
        lines = f.readlines()
    return lines

def save_json(fname, data):
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)
    
def load_json(fname):
    with open(fname, "r") as json_file:
        d = json.load(json_file)
    
    return d

def imread(fname):
    return scipy.misc.imread(fname)


def loadmat(fname):
    return io.loadmat(fname)


@torch.no_grad()
def compute_loss(model, dataset):
    n_images = len(dataset)
    
    loss_sum = 0.
    for i in range(n_images):
        print("{}/{}".format(i, n_images))

        batch = dataset[i]
        batch["images"] = batch["images"][None]
        batch["points"] = batch["points"][None]
        batch["counts"] = batch["counts"][None]
        
        loss_sum += losses.lc_loss(model, batch).item()

    return loss_sum


class RandomSampler(data.sampler.Sampler):
    def __init__(self, train_set):
        self.n_samples = len(train_set)
        self.size = min(self.n_samples, 5000)

    def __iter__(self):
        # np.random.seed(777)
        indices = np.random.randint(0, self.n_samples, self.size)
        # print('indices: ',indices)
        return iter(torch.from_numpy(indices).long())

    def __len__(self):
        return self.size

def poly2mask(rows, cols, shape):
    assert len(rows) == len(cols)
    fill_row_coords, fill_col_coords = draw.polygon(rows, cols, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True

    return mask

def read_xml(fname):
    with open(fname) as f:
        xml = f.readlines()
        xml = ''.join([line.strip('\t') for line in xml])
        
        xml = BeautifulSoup(xml, "html.parser")

    return xml

def load_pkl(fname):
    with open(fname, "rb") as f:        
        return pickle.load(f)


def combine_image_blobs(image_raw, blobs_mask):
    blobs_rgb = label2rgb(blobs_mask)           #different labels with different color

    image_raw = image_raw * 0.5 + blobs_rgb * 0.5
    image_raw /= image_raw.max()

    return mark_boundaries(image_raw, blobs_mask)


def label2rgb(labels):
    labels = np.squeeze(labels)
    colors = color_map(np.max(np.unique(labels)) + 1)
    output = np.zeros(labels.shape + (3,), dtype=np.float64)

    for i in range(len(colors)):
      output[(labels == i).nonzero()] = colors[i]

    return output

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap
