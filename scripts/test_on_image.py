import torch
from src import utils as ut
import torchvision.transforms.functional as FT
from skimage.io import imread,imsave
from torchvision import transforms
from models import model_dict

def apply(image_path, model_name, model_path):
  transformer = ut.ComposeJoint(
                    [
                         [transforms.ToTensor(), None],
                         [transforms.Normalize(*ut.mean_std), None],
                         [None,  ut.ToLong() ]
                    ])  

  # Load best model
  model = model_dict[model_name](n_classes=2).cuda()
  model.load_state_dict(torch.load(model_path))

  # Read Image
  image_raw = imread(image_path)
  collection = list(map(FT.to_pil_image, [image_raw, image_raw]))
  image, _ = transformer(collection)

  batch = {"images":image[None]}
  
  # Make predictions
  pred_blobs = model.predict(batch, method="blobs").squeeze()
  pred_counts = int(model.predict(batch, method="counts").ravel()[0])

  # Save Output
  save_path = image_path + "_blobs_count:{}.png".format(pred_counts)

  imsave(save_path, ut.combine_image_blobs(image_raw, pred_blobs))
  print("| Counts: {}\n| Output saved in: {}".format(pred_counts, save_path))
