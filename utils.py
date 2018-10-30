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

  if metric_name == "MAE_penguin":
    score_dict = val_MAE_penguin(model, dataset, epoch)

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
def val_MAE_penguin(model, dataset, epoch):
  n_images = len(dataset)

  true_count = np.ones(n_images)*(-1)
  pred_count = np.ones(n_images)*(-1)

  true_count_median = np.ones(n_images)*(-1)

  for i in range(n_images):
    batch = dataset[i]
    batch["images"] = batch["images"][None]

    true_count[i] = batch["counts"].item()
    true_count_median[i] = batch["counts_median"].item()
    pred_count[i] = model.predict(batch, method="counts")

    mae = (np.abs(true_count[:i+1] - pred_count[:i+1])).mean() 

    if i % 50 == 0 or i == (n_images - 1):
      print(("%d - %d/%d - Validating %s set - MAE: %.3f" % 
            (epoch, i, n_images, dataset.split, mae)))

  score_dict = {}
  assert not np.any(true_count==(-1))
  assert not np.any(pred_count==(-1))
  score_dict["mae"] = (np.abs(true_count - pred_count)).mean() 
  score_dict["mae_median"] = (np.abs(true_count_median - pred_count)).mean() 

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

def get_experiment(exp_name):
  if exp_name == "trancos":
    dataset_name="trancos"
    model_name="ResFCN"
    metric_name = "MAE"

  if exp_name == "shanghai":
     dataset_name="shanghai"
     model_name="fcn8"
     metric_name = "MAE"

  if exp_name == "pascal":
    dataset_name="pascal"
    model_name="ResFCN"
    metric_name = "mRMSE"

  if exp_name == "penguins":
    dataset_name="penguins"
    model_name="ResFCN"
    metric_name = "MAE_penguin"

  print("Model: {} - Dataset: {} - Metric: {}".format(model_name, dataset_name,metric_name))
  return dataset_name, model_name, metric_name


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
