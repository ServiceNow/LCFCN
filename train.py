import matplotlib
matplotlib.use('Agg')

import os
import torch
import argparse
import losses
import numpy as np
import utils as ut

from torchvision import transforms
from datasets import dataset_dict
from models import model_dict

def train(dataset_name, model_name, metric_name, reset=False):  
  # SET SEED
  np.random.seed(1)
  torch.manual_seed(1) 
  torch.cuda.manual_seed_all(1)

  # Paths
  name = "{}_{}".format(dataset_name, model_name)
  path_model = "checkpoints/model_{}.pth".format(name)
  path_opt = "checkpoints/opt_{}.pth".format(name)
  path_best_model = "checkpoints/best_model_{}.pth".format(name)
  path_history = "checkpoints/history_{}.json".format(name)

  # Train datasets
  transformer = ut.ComposeJoint(
                    [ut.RandomHorizontalFlipJoint(),            
                    [transforms.ToTensor(), None],
                    [transforms.Normalize(*ut.mean_std), None],
                    [None,  ut.ToLong() ]
                    ])

  train_set = dataset_dict[dataset_name](split="train", 
                                         transform_function=transformer)
  trainloader = torch.utils.data.DataLoader(train_set, batch_size=1, 
                                            num_workers=2,
                                            drop_last=False,
                                            sampler=ut.RandomSampler(train_set))
  
  # Val datasets
  transformer = ut.ComposeJoint(
                    [
                         [transforms.ToTensor(), None],
                         [transforms.Normalize(*ut.mean_std), None],
                         [None,  ut.ToLong() ]
                    ])  

  val_set = dataset_dict[dataset_name](split="val", 
                                       transform_function=transformer)

  test_set = dataset_dict[dataset_name](split="test", 
                                       transform_function=transformer)

 

  # Model 
  model = model_dict[model_name](train_set.n_classes).cuda()
  opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=1e-5, weight_decay=0.0005)

  # Train
  if os.path.exists(path_history) and not reset:
    history = ut.load_json(path_history)
    model.load_state_dict(torch.load(path_model))
    opt.load_state_dict(torch.load(path_opt))
    s_epoch = history["train"][-1]["epoch"]
    print("Resuming epoch...{}".format(s_epoch))

  else:
    history = {"train":[], "val":[], "test":[],
               "model_name":model_name,
               "dataset_name":dataset_name, 
               "path_model":path_model,
               "path_opt":path_opt,
               "path_best_model":path_best_model,
               "best_val_epoch":-1, "best_val_mae":np.inf}
    s_epoch = 0
    print("Starting from scratch...")
  

  for epoch in range(s_epoch + 1, 1000):    
    train_dict = ut.fit(model, trainloader, opt, 
                        loss_function=losses.lc_loss,
                        epoch=epoch)
    
    # Update history
    history["trained_images"] = list(model.trained_images)
    history["train"] += [train_dict]

    # Save model, opt and history
    torch.save(model.state_dict(), path_model)
    torch.save(opt.state_dict(), path_opt)
    ut.save_json(path_history, history)

    # %%%%%%%%%%% 2. VALIDATION PHASE %%%%%%%%%%%%"
    with torch.no_grad():
      val_dict = ut.val(model=model, dataset=val_set, epoch=epoch, 
                        metric_name=metric_name)

      # Update history
      history["val"] += [val_dict]

      # Lower is better
      if val_dict[metric_name] <= history["best_val_mae"]:
        history["best_val_epoch"] = epoch
        history["best_val_mae"] = val_dict[metric_name]

        torch.save(model.state_dict(), path_best_model)

        # Test Model
        if not (dataset_name == "penguins" and epoch < 50):
          testDict = ut.val(model=model, 
                                dataset=test_set, 
                                epoch=epoch, metric_name=metric_name)
          history["test"] += [testDict]
        
      ut.save_json(path_history, history)



if __name__ == "__main__":
  # SEE IF CUDA IS AVAILABLE
  assert torch.cuda.is_available()
  print("CUDA: %s" % torch.version.cuda)
  print("Pytroch: %s" % torch.__version__)

  parser = argparse.ArgumentParser()
  parser.add_argument('-r','--reset', default=0, type=int)
  parser.add_argument('-e','--exp_name', default="trancos")
  args = parser.parse_args()

  dataset_name, model_name, metric_name = ut.get_experiment(args.exp_name)
  train(dataset_name=dataset_name, model_name=model_name, 
        metric_name=metric_name, reset=args.reset)