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

# Train Utils
def fit(model, dataloader, opt, loss_function, epoch):
    model.train()

    n_samples = len(dataloader.dataset)
    n_batches = len(dataloader) 
  
    print("Training Epoch {} .... {} batches".format(epoch, n_batches))

    train_dict = {}

    loss_sum = 0.
    for i, batch in enumerate(dataloader):
        opt.zero_grad()
        loss = loss_function(model, batch)
        loss.backward()
        opt.step()

        loss_sum += loss.item()
        if (i % 50) == 0 or i == (n_batches - 1):
            print("{} - ({}/{}) - split: {} - loss: {:.2f}".format(epoch,  i, n_batches, 
                    dataloader.dataset.split, loss_sum/max(1.,i)))
    
    # train
    train_dict["loss"] = loss_sum / n_batches
    train_dict["epoch"] = epoch
    train_dict["n_samples"] = n_samples
    train_dict["iterations"] = n_batches

    return train_dict

# Validation Utils
def val(model, dataset, epoch, metric_name):
  model.eval()
  n_images = len(dataset)

  print("Validating... %d" % n_images)

  if metric_name == "MAE":
    score_dict = val_MAE(model, dataset, epoch)

  elif metric_name == "mRMSE":
    score_dict = val_mRMSE(model, dataset, epoch)

  score_dict["n_samples"] = n_images
  score_dict["epoch"] = epoch  
  score_dict["split_name"] = dataset.split

  return score_dict

@torch.no_grad()
def val_MAE(model, dataset, epoch):
  n_images = len(dataset)

  true_count = np.ones(n_images)*(-1)
  pred_count = np.ones(n_images)*(-1)

  for i in range(n_images):
    batch = dataset[i]
    batch["images"] = batch["images"][None]
    
    # Make sure model wasn't trained on this
    assert batch["image_path"] not in model.trained_images
    # print("model wasn't trained on this!")
    
    true_count[i] = batch["counts"].item()
    pred_count[i] = model.predict(batch, method="counts")

    mae = (np.abs(true_count[:i+1] - pred_count[:i+1])).mean() 

    if i % 50 == 0 or i == (n_images - 1):
      print(("%d - %d/%d - Validating %s set - MAE: %.3f" % 
            (epoch, i, n_images, dataset.split, mae)))

  score_dict = {}
  assert not np.any(true_count==(-1))
  assert not np.any(pred_count==(-1))
  score_dict["MAE"] = (np.abs(true_count - pred_count)).mean() 

  return score_dict


@torch.no_grad()
def val_mRMSE(model, dataset, epoch):
  n_images = len(dataset)
  true_count = np.ones((n_images,20))*(-1)
  pred_count = np.ones((n_images,20))*(-1)

  for i in range(n_images):
    batch = dataset[i]
    batch["images"] = batch["images"][None]

    assert batch["image_path"] not in model.trained_images

    true_count[i] = t2n(batch["counts"])
    pred_count[i] = model.predict(batch, method="counts")


    mRMSE = np.sqrt(np.mean((pred_count[:i+1] - true_count[:i+1])**2, 0)).mean()

    if i % 50 == 0 or i == (n_images - 1):
      print(("%d - %d/%d - Validating %s set - mRMSE: %.3f" % 
            (epoch, i, n_images, dataset.split, mRMSE)))

  score_dict = {}
  assert not np.any(true_count==(-1))
  assert not np.any(pred_count==(-1))
  score_dict["mRMSE"] = np.sqrt(np.mean((pred_count - true_count)**2, 0)).mean()

  return score_dict




# Transforms Utils
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class ComposeJoint(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = self._iterate_transforms(transform, x)
            
        return x
    
    def _iterate_transforms(self, transforms, x):
        if isinstance(transforms, collections.Iterable):
            for i, transform in enumerate(transforms):
                x[i] = self._iterate_transforms(transform, x[i])
        else:
            
            if transforms is not None:
                x = transforms(x)
                
        return x

class RandomHorizontalFlipJoint(object):
    def __call__(self, inputs):
        # Perform the same flip on all of the inputs
        if random.random() < 0.5:
            return list(map(lambda single_input:  
                    ImageOps.mirror(single_input), inputs))
        
        
        return inputs

class ToLong(object):
    def __call__(self, x):
        return torch.LongTensor(np.asarray(x))

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
        indices =  np.random.randint(0, self.n_samples, self.size)
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
        
        xml = BeautifulSoup(xml, "lxml")

    return xml

def load_pkl(fname):
    with open(fname, "rb") as f:        
        return pickle.load(f)


def combine_image_blobs(image_raw, blobs_mask):
  blobs_rgb = label2rgb(blobs_mask)

  image_raw = image_raw*0.5 + blobs_rgb * 0.5
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