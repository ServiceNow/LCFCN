import torch
import skimage
import torch.nn.functional as F
import numpy as np
from skimage.morphology import watershed
from skimage.segmentation import find_boundaries
from scipy import ndimage
from skimage import morphology as morph
from skimage.morphology import watershed
from skimage.segmentation import find_boundaries


def compute_lcfcn_loss(logits, points, reduction='mean',
                       add_global_split=True, 
                       roi_mask=None):
    """Computes the lcfcn loss.

    Parameters
    ----------
    logits : torch tensor of shape (n, c, h ,w)
        Model output before the softmax
    points : torch tensor of shape (n, h ,w)
        Non-zero entries represent the locations and the class id of the objects

    Returns
    -------
    loss: float
        LCFCN loss
    """
    n = logits.size(0)
    assert n == 1

    probs = F.softmax(logits, 1)
    probs_log = F.log_softmax(logits, 1)

    # IMAGE LOSS
    iloss = compute_image_loss(probs, points)

    # POINT LOSS
    ploss = F.nll_loss(probs_log, points,
                       ignore_index=0,
                       reduction='sum')

    blob_dict = get_blob_dict(logits, points, roi_mask=roi_mask)

    # split loss
    sloss = compute_split_loss(probs_log, probs,
                                   points, blob_dict, 
                                   add_global_loss=add_global_split, 
                                   reduction=reduction)

    # FP loss
    floss = compute_fp_loss(probs_log, blob_dict, 
                                reduction=reduction)



    return iloss + ploss + sloss + floss

# Loss Utils


def compute_image_loss(logits, points):
    n, k, h, w = logits.size()

    # get target
    labels = torch.zeros(k, device=logits.device)
    labels[points.unique()] = 1
    logits_max = logits.view(n, k, h*w).max(2)[0].view(-1)

    loss = F.binary_cross_entropy(logits_max, labels, reduction='sum')

    return loss


def compute_fp_loss(probs_log, blob_dict, reduction='sum'):
    if blob_dict["n_fp"] == 0:
        return 0.

    blobs = blob_dict["blobs"]

    
    loss = 0.
    n_fp = 0.
    for b in blob_dict["blobList"]:
        if b["n_points"] != 0:
            continue

        T = np.ones(blobs.shape[-2:])
        T[blobs[b["class"]] == b["label"]] = 0

        loss += F.nll_loss(probs_log, torch.LongTensor(T).cuda()[None],
                                   ignore_index=1, reduction='mean')
        n_fp += 1

    if reduction == 'mean':
        loss = loss / max(n_fp, 1)

    return loss


def compute_split_loss(probs_log, probs, points, blob_dict,
                 add_global_loss=False, reduction='sum'):
    if blob_dict["n_multi"] == 0:
        return 0.

    blobs = blob_dict["blobs"]
    probs_numpy = probs[0].detach().cpu().numpy()
    points_numpy = points.cpu().numpy().squeeze()

    loss = 0.
    n_multi = 0.
    for b in blob_dict["blobList"]:
        if b["n_points"] < 2:
            continue

        l = b["class"] + 1
        probs = probs_numpy[b["class"] + 1]

        points_class = (points_numpy == l).astype("int")
        blob_ind = blobs[b["class"]] == b["label"]

        T = watersplit(probs, points_class*blob_ind)*blob_ind
        T = 1 - T

        loss += (b["n_points"] - 1) * F.nll_loss(probs_log, torch.LongTensor(T).cuda()[None],
                                          ignore_index=1, reduction='mean')
        n_multi += 1
        
    if reduction == 'mean':
        loss = loss / max(n_multi, 1)

    if add_global_loss:
        for l in range(1, probs_numpy.shape[1]):
            points_class = (points_numpy == l).astype(int)

            if points_class.sum() == 0:
                continue

            T = watersplit(probs_numpy[l], points_class)
            T = 1 - T
            scale = float(points_class.sum())
            # hu.save_image('tmp.png', T)
            loss += scale * F.nll_loss(probs_log, torch.LongTensor(T).cuda()[None],
                               ignore_index=1, reduction='mean')

    return loss


def watersplit(_probs, _points):
    points = _points.copy()

    points[points != 0] = np.arange(1, points.sum()+1)
    points = points.astype(float)

    probs = ndimage.black_tophat(_probs.copy(), 7)
    seg = watershed(probs, points)

    return find_boundaries(seg)


def get_blobs(logits, roi_mask=None):
    n, k, _, _ = logits.shape
    pred_mask = logits.max(1)[1].squeeze().cpu().numpy()
    
    h, w = pred_mask.shape
    blobs = np.zeros((k - 1, h, w), int)

    for category_id in np.unique(pred_mask):
        if category_id == 0:
            continue
        blobs[category_id - 1] = morph.label(pred_mask == category_id)

    if roi_mask is not None:
        blobs = (blobs * roi_mask[None]).astype(int)

    return blobs


@torch.no_grad()
def get_blob_dict(logits, points, roi_mask=None):
    blobs = get_blobs(logits, roi_mask=roi_mask)
    points = points.cpu().numpy().squeeze()

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
        blob_uniques, blob_counts = np.unique(
            class_blobs * (points_mask), return_counts=True)
        uniques = np.delete(np.unique(class_blobs), blob_uniques)

        for u in uniques:
            blobList += [{"class": l, 
                          "label": u, 
                          "n_points": 0, 
                          "size": 0,
                          "pointsList": []}]
            n_fp += 1

        for i, u in enumerate(blob_uniques):
            if u == 0:
                continue

            pointsList = []
            blob_ind = class_blobs == u

            locs = np.where(blob_ind * (points_mask))

            for j in range(locs[0].shape[0]):
                pointsList += [{"y": locs[0][j], "x":locs[1][j]}]

            assert len(pointsList) == blob_counts[i]

            if blob_counts[i] == 1:
                n_single += 1

            else:
                n_multi += 1
            size = blob_ind.sum()
            total_size += size
            blobList += [{"class": l, "size": size,
                          "label": u, "n_points": blob_counts[i],
                          "pointsList":pointsList}]

    blob_dict = {"blobs": blobs, "blobList": blobList,
                 "n_fp": n_fp,
                 "n_single": n_single,
                 "n_multi": n_multi,
                 "total_size": total_size}

    return blob_dict


def blobs2points(blobs):
    blobs = blobs.squeeze()
    points = np.zeros(blobs.shape).astype("uint8")
    rps = skimage.measure.regionprops(blobs)

    assert points.ndim == 2

    for r in rps:
        y, x = r.centroid

        points[int(y), int(x)] = 1

    return points

def compute_game(pred_points, gt_points, L=1):
    n_rows = 2**L
    n_cols = 2**L

    pred_points = pred_points.astype(float).squeeze()
    gt_points = np.array(gt_points).astype(float).squeeze()
    h, w = pred_points.shape
    se = 0.

    hs, ws = h//n_rows, w//n_cols
    for i in range(n_rows):
        for j in range(n_cols):

            sr, er = hs*i, hs*(i+1)
            sc, ec = ws*j, ws*(j+1)

            pred_count = pred_points[sr:er, sc:ec]
            gt_count = gt_points[sr:er, sc:ec]
            
            se += float(abs(gt_count.sum() - pred_count.sum()))
    return se

def save_tmp(fname, images, logits, points):
    from haven import haven_utils as hu
    probs = F.softmax(logits, 1); 
    mask = probs.argmax(dim=1).cpu().numpy().astype('uint8').squeeze()*255
    img_mask=hu.save_image('tmp2.png', 
                hu.denormalize(images, mode='rgb'), 
                mask=mask, return_image=True)
    hu.save_image(fname,np.array(img_mask)/255. , 
                    points=att_dict['points'])
