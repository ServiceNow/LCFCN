import torch
import torch.nn.functional as F
import numpy as np 
from skimage.morphology import watershed
from skimage.segmentation import find_boundaries
from scipy import ndimage
import utils as ut

def lc_loss(model, batch):
    
    model.train()
    N =  batch["images"].size(0)
    assert N == 1

    blob_dict = get_blob_dict(model, batch)
    # put variables in cuda
    images = batch["images"].cuda()
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()
    #print(images.shape)

    O = model(images)
    S = F.softmax(O, 1)
    S_log = F.log_softmax(O, 1)

    # IMAGE LOSS
    loss = compute_image_loss(S, counts)

    # POINT LOSS
    loss += F.nll_loss(S_log, points, 
                       ignore_index=0,
                       reduction='sum')
    # FP loss
    if blob_dict["n_fp"] > 0:
        loss += compute_fp_loss(S_log, blob_dict)

    # split_mode loss
    if blob_dict["n_multi"] > 0:
        loss += compute_split_loss(S_log, S, points, blob_dict)

    # Global loss 
    S_npy = ut.t2n(S.squeeze())
    points_npy = ut.t2n(points).squeeze()
    for l in range(1, S.shape[1]):
        points_class = (points_npy==l).astype(int)

        if points_class.sum() == 0:
            continue

        T = watersplit(S_npy[l], points_class)
        T = 1 - T
        scale = float(counts.sum())
        loss += float(scale) * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                        ignore_index=1, reduction='elementwise_mean')

    # Add to trained images
    model.trained_images.add(batch["image_path"][0])

    return loss / N


# Loss Utils
def compute_image_loss(S, Counts):
    n,k,h,w = S.size()

    # GET TARGET
    ones = torch.ones(Counts.size(0), 1).long().cuda()
    BgFgCounts = torch.cat([ones, Counts], 1)
    Target = (BgFgCounts.view(n*k).view(-1) > 0).view(-1).float()

    # GET INPUT
    Smax = S.view(n, k, h*w).max(2)[0].view(-1)
    
    loss = F.binary_cross_entropy(Smax, Target, reduction='sum')
    
    return loss

def compute_fp_loss(S_log, blob_dict):
    blobs = blob_dict["blobs"]
    
    scale = 1.
    loss = 0.

    for b in blob_dict["blobList"]:
        if b["n_points"] != 0:
            continue

        T = np.ones(blobs.shape[-2:])
        T[blobs[b["class"]] == b["label"]] = 0

        loss += scale * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                        ignore_index=1, reduction='elementwise_mean')
    return loss 

def compute_split_loss(S_log, S, points, blob_dict):
    blobs = blob_dict["blobs"]
    S_numpy = ut.t2n(S[0])
    points_numpy = ut.t2n(points).squeeze() 

    loss = 0.

    for b in blob_dict["blobList"]:
        if b["n_points"] < 2:
            continue

        l = b["class"] + 1
        probs = S_numpy[b["class"] + 1]

        points_class = (points_numpy==l).astype("int")
        blob_ind = blobs[b["class"] ] == b["label"]

        T = watersplit(probs, points_class*blob_ind)*blob_ind
        T = 1 - T

        scale = b["n_points"] + 1
        loss += float(scale) * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                        ignore_index=1, reduction='elementwise_mean')

    return loss 


def watersplit(_probs, _points):
   points = _points.copy()

   points[points!=0] = np.arange(1, points.sum()+1)
   points = points.astype(float)

   probs = ndimage.black_tophat(_probs.copy(), 7)   
   seg =  watershed(probs, points)

   return find_boundaries(seg)


@torch.no_grad()
def get_blob_dict(model, batch, training=False): 
    blobs = model.predict(batch, method="blobs").squeeze()
    points = ut.t2n(batch["points"]).squeeze()

    if blobs.ndim == 2:
        blobs = blobs[None]

    blobList = []

    n_multi = 0
    n_single = 0
    n_fp = 0
    total_size = 0

    for l in range(blobs.shape[0]):
        class_blobs = blobs[l]
        points_mask = points == (l+1)
        # Intersecting
        blob_uniques, blob_counts = np.unique(class_blobs * (points_mask), return_counts=True)
        uniques = np.delete(np.unique(class_blobs), blob_uniques)

        for u in uniques:
            blobList += [{"class":l, "label":u, "n_points":0, "size":0,
                         "pointsList":[]}]
            n_fp += 1

        for i, u in enumerate(blob_uniques):
            if u == 0:
                continue

            pointsList = []
            blob_ind = class_blobs==u

            locs = np.where(blob_ind * (points_mask))

            for j in range(locs[0].shape[0]):
                pointsList += [{"y":locs[0][j], "x":locs[1][j]}]
            
            assert len(pointsList) == blob_counts[i]

            if blob_counts[i] == 1:
                n_single += 1

            else:
                n_multi += 1
            size = blob_ind.sum()
            total_size += size
            blobList += [{"class":l, "size":size, 
                          "label":u, "n_points":blob_counts[i],
                          "pointsList":pointsList}]

    blob_dict = {"blobs":blobs, "blobList":blobList, 
                 "n_fp":n_fp, 
                 "n_single":n_single,
                 "n_multi":n_multi,
                 "total_size":total_size}

    return blob_dict