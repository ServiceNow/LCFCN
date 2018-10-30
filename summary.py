import matplotlib
matplotlib.use('Agg')

import pandas as pd
import utils as ut

def summary(dataset_name, model_name, 
            path_history="checkpoints/history.json"):
            # checkpoints_path="/mnt/home/issam/LCFCNSaves/"):
  
  history = ut.load_json(path_history)

  print("\nTrain-----------")
  print(pd.DataFrame(history["train"]))
  
  print("\nVAL-----------")
  print(pd.DataFrame(history["val"]))
  print("\nBEST VAL-----------")
  print(pd.DataFrame([v for v in history["val"] if v["epoch"] == history["best_val_epoch"] ]))
  
  print("\nTEST----------")
  print(pd.DataFrame(history["test"][-1:]).tail())
