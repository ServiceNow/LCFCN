import time

import torch
import unittest
import numpy as np 
import os, sys
import torch
import shutil, time
import copy

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)
from lcfcn import lcfcn_loss

if __name__ == "__main__":
    n, c, h, w = 1, 3, 100, 100
    prob =  0.5
    for device in ['cpu', 'cuda']:
        points = (torch.FloatTensor(n, h, w).uniform_() > prob).long().to(device=device)
        n_points = int(points.sum())
        logits = torch.randn(n, c, h, w).to(device=device)

        n_times = 4
        s_time = time.time()
        for i in range(n_times):
            loss = lcfcn_loss.compute_lcfcn_loss(logits, points)
            # print(loss)
        print('\n%s Time for (%d, %d) images with %d points: %.3f seconds' % 
                (device, h,w,n_points, (time.time() - s_time) / n_times))