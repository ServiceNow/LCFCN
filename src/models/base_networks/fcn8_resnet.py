import torch.nn as nn
import torchvision
import torch
from skimage import morphology as morph
import numpy as np
from torch import optim
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import numpy as np
from skimage.morphology import watershed
from skimage.segmentation import find_boundaries
from scipy import ndimage
 

class FCN8(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_classes = 2

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_32s = torchvision.models.resnet50(pretrained=True)
        resnet_block_expansion_rate = resnet50_32s.layer1[0].expansion

        # Create a linear layer -- we don't need logits in this case
        resnet50_32s.fc = nn.Sequential()

        self.resnet50_32s = resnet50_32s

        self.score_32s = nn.Conv2d(512 * resnet_block_expansion_rate,
                                   self.n_classes,
                                   kernel_size=1)

        self.score_16s = nn.Conv2d(256 * resnet_block_expansion_rate,
                                   self.n_classes,
                                   kernel_size=1)

        self.score_8s = nn.Conv2d(128 * resnet_block_expansion_rate,
                                  self.n_classes,
                                  kernel_size=1)


        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False


    def extract_features(self, x_input):
        self.resnet50_32s.eval()
        x = self.resnet50_32s.conv1(x_input)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x = self.resnet50_32s.maxpool(x)

        x = self.resnet50_32s.layer1(x)
        
        x_8s = self.resnet50_32s.layer2(x)
        x_16s = self.resnet50_32s.layer3(x_8s)
        x_32s = self.resnet50_32s.layer4(x_16s)

        return x_8s, x_16s, x_32s

    

    def forward(self, x):
        self.resnet50_32s.eval()
        input_spatial_dim = x.size()[2:]

        x = self.resnet50_32s.conv1(x)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x = self.resnet50_32s.maxpool(x)

        x = self.resnet50_32s.layer1(x)

        x = self.resnet50_32s.layer2(x)
        logits_8s = self.score_8s(x)

        x = self.resnet50_32s.layer3(x)
        logits_16s = self.score_16s(x)

        x = self.resnet50_32s.layer4(x)
        logits_32s = self.score_32s(x)

        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]

        logits_16s += nn.functional.interpolate(logits_32s,
                                                             size=logits_16s_spatial_dim,
                                                             mode="bilinear",
                                                             align_corners=True)

        logits_8s += nn.functional.interpolate(logits_16s,
                                                            size=logits_8s_spatial_dim,
                                                            mode="bilinear",
                                                            align_corners=True)

        logits_upsampled = nn.functional.interpolate(logits_8s,
                                                                  size=input_spatial_dim,
                                                                  mode="bilinear",
                                                                  align_corners=True)

        return logits_upsampled

