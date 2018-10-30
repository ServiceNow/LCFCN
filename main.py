import matplotlib
matplotlib.use('Agg')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"

import torch
import argparse
import numpy as np
import train, test, summary
import utils as ut

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-e','--exp_name', default="trancos")
  parser.add_argument('-m','--mode', default="summary")
  args = parser.parse_args()

  dataset_name, model_name, metric_name = ut.get_experiment(args.exp_name)

  # Paths
  name = "{}_{}".format(dataset_name, model_name)
  
  path_model = "checkpoints/model_{}.pth".format(name)
  path_opt = "checkpoints/opt_{}.pth".format(name)
  path_best_model = "checkpoints/best_model_{}.pth".format(name)
  path_history = "checkpoints/history_{}.json".format(name)


  if args.mode == "train":
    train.train(dataset_name, model_name, metric_name, path_history, path_model, path_opt, path_best_model,)
  
  if args.mode == "test":
    test.test(dataset_name, model_name, metric_name, path_history, path_best_model)

  if args.mode == "summary":      
    summary.summary(dataset_name, model_name, path_history)
        
if __name__ == "__main__":
    main()