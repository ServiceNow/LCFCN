import torch
from torchvision import transforms
import pandas as pd
from skimage.segmentation import expand_labels
from skimage.color import label2rgb
import os, tqdm
from skimage import segmentation
import numpy as np
from lcfcn import lcfcn_loss
from haven import haven_utils as hu
from . import networks


class LCFCN(torch.nn.Module):
    def __init__(self, n_classes=1, lr=1e-5, opt='adam', network='vgg', device='cuda'):
        super().__init__()
        self.device = device
        if network == 'vgg':
            self.model_base = networks.FCN8_VGG16(n_classes=n_classes)
        elif network == 'resnet':
            self.model_base = networks.FCN8_ResNet(n_classes=n_classes)

        if opt == "adam":
            self.opt = torch.optim.Adam(
                self.model_base.parameters(), lr=lr, betas=(0.99, 0.999), weight_decay=0.0005)

        elif opt == "sgd":
            self.opt = torch.optim.SGD(
                self.model_base.parameters(), lr=lr)

        else:
            raise ValueError

    def train_on_loader(model, train_loader):
        model.train()

        n_batches = len(train_loader)
        
        pbar = tqdm.tqdm(total=n_batches)
        for batch in train_loader:
            score_dict = model.train_on_batch(batch)

            pbar.set_description("Training. Loss: %.4f" % score_dict['train_loss'])
            pbar.update(1)

        pbar.close()

        return {'train_loss':score_dict['train_loss']}

    @torch.no_grad()
    def val_on_loader(self, val_loader, savedir_images=None, n_images=2):
        self.eval()

        n_batches = len(val_loader)
        score_list = []
        pbar = tqdm.tqdm(total=n_batches)
        for i, batch in enumerate(tqdm.tqdm(val_loader)):
            val_dict = self.val_on_batch(batch)
            score_list += [val_dict]
            
            pbar.update(1)

            if savedir_images and i < n_images:
                os.makedirs(savedir_images, exist_ok=True)
                self.vis_on_batch(batch, savedir_image=os.path.join(
                    savedir_images, "%d.jpg" % i))
                
                pbar.set_description("Validating. MAE: %.4f" % val_dict['mae'])

        pbar.close()
        val_dict =pd.DataFrame(score_list).mean().to_dict()
        return val_dict

    def train_on_batch(self, batch, **extras):
        self.opt.zero_grad()
        self.train()

        images = batch["images"].to(self.device)
        points = batch["points"].long().to(self.device)
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
        
    @torch.no_grad()
    def val_on_batch(self, batch):
        self.eval()
        images = batch["images"].to(self.device)
        points = batch["points"].long().to(self.device)
        logits = self.model_base.forward(images)
        probs = logits.sigmoid().cpu().numpy()

        blobs = lcfcn_loss.get_blobs(probs=probs)
        pred_points = lcfcn_loss.blobs2points(blobs).squeeze()

        mae = abs(float((np.unique(blobs)!=0).sum() - (points!=0).sum()))
        game = lcfcn_loss.compute_game(pred_points=pred_points.squeeze(), gt_points=points.squeeze().cpu().numpy(), L=3)
        return {'mae':mae, 'game':game }
        
    @torch.no_grad()
    def vis_on_batch(self, batch, savedir_image):
        self.eval()
        images = batch["images"].to(self.device)
        points = batch["points"].long().to(self.device)
        logits = self.model_base.forward(images)
        probs = logits.sigmoid().cpu().numpy()

        blobs = lcfcn_loss.get_blobs(probs=probs)

        pred_counts = (np.unique(blobs)!=0).sum()
        pred_blobs = blobs
        pred_probs = probs.squeeze()

        # loc 
        pred_count = pred_counts.ravel()[0]
        pred_blobs = pred_blobs.squeeze()
        pred_points = lcfcn_loss.blobs2points(pred_blobs).squeeze()
        img_org = hu.get_image(batch["images"],denorm="rgb")

        i1 = convert(np.array(img_org), batch['points'][0], enlarge=20)
        i2 = convert(np.array(img_org), pred_blobs, enlarge=0)
        i3 = convert(np.array(img_org), pred_points, enlarge=20)
        
        
        hu.save_image(savedir_image, np.hstack([i1, i2, i3]))
     

def convert(img, mask, enlarge=0):
    if enlarge != 0:
        mask = expand_labels(mask, enlarge).astype('uint8')
    m = label2rgb(mask, bg_label=0)
    m = segmentation.mark_boundaries(m, mask.astype('uint8'))
    i = 0.5 * np.array(img) / 255.
    ind = m != 0
    i[ind] = m[ind] 

    return i

def transform_image(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    return transform(image)
