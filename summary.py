import matplotlib
matplotlib.use('Agg')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"

import pandas as pd
import os
import torch
import argparse
import losses
import numpy as np
import utils as ut

from torchvision import transforms
from datasets import dataset_dict
from models import model_dict

def summary(dataset_name, model_name):
  path_history = "checkpoints/history_{}_{}.json".format(dataset_name, model_name)
  history = ut.load_json(path_history)

  print("\nTrain-----------")
  print(pd.DataFrame(history["train"]))

  print("\nVAL-----------")
  print(pd.DataFrame(history["val"]))
  print("\nBEST VAL-----------")
  print(pd.DataFrame([v for v in history["val"] if v["epoch"] == history["best_val_epoch"] ]))
  
  print("\nTEST----------")
  print(pd.DataFrame(history["test"][-1:]).tail())

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-e','--exp_name', default="trancos")
  args = parser.parse_args()

  dataset_name, model_name, metric_name = ut.get_experiment(args.exp_name)
  summary(dataset_name=dataset_name, model_name=model_name)