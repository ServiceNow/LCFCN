import matplotlib
matplotlib.use('Agg')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"

import argparse
import applyOnImage
import experiments, train, test, summary

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-e','--exp_name', default="trancos")
  parser.add_argument('-m','--mode', default="summary")
  parser.add_argument('-image','--image_path', default="summary")
  args = parser.parse_args()

  dataset_name, model_name, metric_name = experiments.get_experiment(args.exp_name)

  # Paths
  name = "{}_{}".format(dataset_name, model_name)
  
  path_model = "checkpoints/model_{}.pth".format(name)
  path_opt = "checkpoints/opt_{}.pth".format(name)
  path_best_model = "checkpoints/best_model_{}.pth".format(name)
  path_history = "checkpoints/history_{}.json".format(name)


  if args.mode == "train":
    train.train(dataset_name, model_name, metric_name, path_history, path_model, path_opt, path_best_model,)

  if args.mode == "apply":    
    applyOnImage.apply(args.image_path, model_name, path_best_model)

  if args.mode == "test":
    test.test(dataset_name, model_name, metric_name, path_history, path_best_model)

  if args.mode == "summary":      
    summary.summary(dataset_name, model_name, path_history)
        
if __name__ == "__main__":
    main()