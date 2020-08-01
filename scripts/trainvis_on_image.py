import sys, os
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)

import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import exp_configs
import time
import numpy as np

from src import models
from src import datasets


import argparse

from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src import datasets
# from src import optimizers 
import torchvision

cudnn.benchmark = True

from haven import haven_utils as hu
from haven import haven_img as hi
from haven import haven_results as hr
from haven import haven_chk as hc
# from src import looc_utils as lu
from PIL import Image



if __name__ == "__main__":
    exp_dict = {"dataset": {'name':'trancos', 
                          'transform':'rgb_normalize'},
         "model": {'name':'lcfcn','base':"fcn8_vgg16"},
         "batch_size": 1,
         "max_epoch": 100,
         'dataset_size': {'train':1, 'val':1},
         'optimizer':'adam',
         'lr':1e-5}
    
    train_set = datasets.get_dataset(dataset_dict=exp_dict['dataset'],
                datadir='/mnt/public/datasets/Trancos', split="test",exp_dict=exp_dict)
    model = models.get_model(model_dict=exp_dict['model'],
                             exp_dict=exp_dict,
                             train_set=train_set).cuda()
    batch = train_set[0]
    batch['images'] = batch['images'][None]
    batch['points'] = batch['points'][None]

    # train for several iterations
    for i in range(1000):
        loss = model.train_on_batch(batch)
        print(i, '- loss:', float(loss['train_loss']))

    # visualize blobs and heatmap
    model.vis_on_batch(batch, savedir_image='result.png')