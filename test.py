import matplotlib
matplotlib.use('Agg')

import torch
import argparse
import utils as ut
import pandas as pd 

from torchvision import transforms
from datasets import dataset_dict
from models import model_dict

def test(dataset_name, model_name, metric_name, 
         checkpoints_path="checkpoints/"):

  path_history = "{}/{}_{}/history.json".format(checkpoints_path, dataset_name, model_name)
  path_best_model = "{}/{}_{}/State_Dicts/best_model.pth".format(checkpoints_path, dataset_name, model_name)

  history = ut.load_json(path_history)

  transformer = ut.ComposeJoint(
                    [
                         [transforms.ToTensor(), None],
                         [transforms.Normalize(*ut.mean_std), None],
                         [None,  ut.ToLong() ]
                    ])  
  test_set = dataset_dict[dataset_name](split="test", 
                                       transform_function=transformer)

  model = model_dict[model_name](n_classes=test_set.n_classes).cuda()
  # path_best_model = "/mnt/home/issam/LCFCNSaves/pascal/State_Dicts/best_model.pth"
  model.load_state_dict(torch.load(path_best_model))
  
  testDict = ut.val(model=model, dataset=test_set, 
                    epoch=history["best_val_epoch"], metric_name=metric_name)

  print(pd.DataFrame([testDict]))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-e','--exp_name', default="trancos")
  args = parser.parse_args()

  dataset_name, model_name, metric_name = ut.get_experiment(args.exp_name)
  test(dataset_name=dataset_name, model_name=model_name,metric_name=metric_name)