import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import os, tqdm
import numpy as np
import time
from sklearn.metrics import confusion_matrix
import skimage
from lcfcn import lcfcn_loss
from src import models
from haven import haven_img as hi
from scipy import ndimage
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import cv2
from haven import haven_img
from haven import haven_utils as hu
from . import base_networks, metrics


class LCFCN(torch.nn.Module):
    def __init__(self, exp_dict, train_set):
        super().__init__()
        self.exp_dict = exp_dict
        self.n_classes = train_set.n_classes
        self.exp_dict = exp_dict

        self.model_base = base_networks.get_base(self.exp_dict['model']['base'],
                                               self.exp_dict, n_classes=self.n_classes)

        if self.exp_dict["optimizer"] == "adam":
            self.opt = torch.optim.Adam(
                self.model_base.parameters(), lr=self.exp_dict["lr"], betas=(0.99, 0.999), weight_decay=0.0005)

        elif self.exp_dict["optimizer"] == "sgd":
            self.opt = torch.optim.SGD(
                self.model_base.parameters(), lr=self.exp_dict["lr"])

        else:
            raise ValueError

    def train_on_loader(model, train_loader):
        model.train()

        n_batches = len(train_loader)
        train_meter = metrics.Meter()
        
        pbar = tqdm.tqdm(total=n_batches)
        for batch in train_loader:
            score_dict = model.train_on_batch(batch)
            train_meter.add(score_dict['train_loss'], batch['images'].shape[0])

            pbar.set_description("Training. Loss: %.4f" % train_meter.get_avg_score())
            pbar.update(1)

        pbar.close()

        return {'train_loss':train_meter.get_avg_score()}

    @torch.no_grad()
    def val_on_loader(self, val_loader, savedir_images=None, n_images=2):
        self.eval()

        n_batches = len(val_loader)
        val_meter = metrics.Meter()
        pbar = tqdm.tqdm(total=n_batches)
        for i, batch in enumerate(tqdm.tqdm(val_loader)):
            score_dict = self.val_on_batch(batch)
            val_meter.add(score_dict['miscounts'], batch['images'].shape[0])
            
            pbar.update(1)

            if savedir_images and i < n_images:
                os.makedirs(savedir_images, exist_ok=True)
                self.vis_on_batch(batch, savedir_image=os.path.join(
                    savedir_images, "%d.jpg" % i))
                
                pbar.set_description("Validating. MAE: %.4f" % val_meter.get_avg_score())

        pbar.close()
        val_mae = val_meter.get_avg_score()
        val_dict = {'val_mae':val_mae, 'val_score':-val_mae}
        return val_dict

    def train_on_batch(self, batch, **extras):
        self.opt.zero_grad()
        self.train()

        images = batch["images"].cuda()
        points = batch["points"].long().cuda()
        logits = self.model_base.forward(images)
        loss = lcfcn_loss.compute_loss(points=points, probs=logits.sigmoid())
        
        loss.backward()

        self.opt.step()

        return {"train_loss":loss.item()}


    def get_state_dict(self):
        state_dict = {"model": self.model_base.state_dict(),
                      "opt":self.opt.state_dict()}

        return state_dict

    def load_state_dict(self, state_dict):
        self.model_base.load_state_dict(state_dict["model"])
        self.opt.load_state_dict(state_dict["opt"])

    def val_on_batch(self, batch):
        self.eval()
        images = batch["images"].cuda()
        points = batch["points"].long().cuda()
        logits = self.model_base.forward(images)
        probs = logits.sigmoid().cpu().numpy()

        blobs = lcfcn_loss.get_blobs(probs=probs)

        return {'miscounts': abs(float((np.unique(blobs)!=0).sum() - 
                                (points!=0).sum()))}
        
    @torch.no_grad()
    def vis_on_batch(self, batch, savedir_image):
        self.eval()
        images = batch["images"].cuda()
        points = batch["points"].long().cuda()
        logits = self.model_base.forward(images)
        probs = logits.sigmoid().cpu().numpy()

        blobs = lcfcn_loss.get_blobs(probs=probs)

        pred_counts = (np.unique(blobs)!=0).sum()
        pred_blobs = blobs
        pred_probs = probs.squeeze()

        # loc 
        pred_count = pred_counts.ravel()[0]
        pred_blobs = pred_blobs.squeeze()
        
        img_org = hu.get_image(batch["images"],denorm="rgb")

        # true points
        y_list, x_list = np.where(batch["points"][0].long().numpy().squeeze())
        img_peaks = haven_img.points_on_image(y_list, x_list, img_org)
        text = "%s ground truth" % (batch["points"].sum().item())
        haven_img.text_on_image(text=text, image=img_peaks)

        # pred points 
        pred_points = lcfcn_loss.blobs2points(pred_blobs).squeeze()
        y_list, x_list = np.where(pred_points.squeeze())
        img_pred = hi.mask_on_image(img_org, pred_blobs)
        # img_pred = haven_img.points_on_image(y_list, x_list, img_org)
        text = "%s predicted" % (len(y_list))
        haven_img.text_on_image(text=text, image=img_pred)

        # heatmap 
        heatmap = hi.gray2cmap(pred_probs)
        heatmap = hu.f2l(heatmap)
        haven_img.text_on_image(text="lcfcn heatmap", image=heatmap)
        
        
        img_mask = np.hstack([img_peaks, img_pred, heatmap])
        
        hu.save_image(savedir_image, img_mask)
     