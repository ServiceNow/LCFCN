import matplotlib
matplotlib.use('Agg')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"

import pandas as pd
import os
import argparse
import utils as ut

def summary(dataset_name, model_name, 
            checkpoints_path="checkpoints/"):
            # checkpoints_path="/mnt/home/issam/LCFCNSaves/"):
  
  path_history = "{}/{}_{}/history.json".format(checkpoints_path, dataset_name, model_name)
  history = ut.load_json(path_history)

  # if 1:
  #   epoch = 695
  #   path_test = "{}/{}_{}/test_MAE_695.json".format(checkpoints_path, dataset_name, model_name)
  #   test_dict = ut.load_json(path_test)
  #   test_dict["epoch"] = epoch
  #   history["test"] = [test_dict]
  #   history["best_val_epoch"] = epoch
  #   ut.save_json(path_history, history)

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