
import torch.nn as nn
import torch
import numpy as np
import torchvision
import torch.utils.model_zoo as model_zoo


class FCN8_VGG16(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        # PREDEFINE LAYERS
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.relu = nn.ReLU(inplace=True)
      
        # VGG16 PART
        self.conv1_1 = conv3x3(3, 64, stride=1, padding=100)
        self.conv1_2 = conv3x3(64, 64)
        
        self.conv2_1 = conv3x3(64, 128)
        self.conv2_2 = conv3x3(128, 128)
        
        self.conv3_1 = conv3x3(128, 256)
        self.conv3_2 = conv3x3(256, 256)
        self.conv3_3 = conv3x3(256, 256)

        self.conv4_1 = conv3x3(256, 512)
        self.conv4_2 = conv3x3(512, 512)
        self.conv4_3 = conv3x3(512, 512)

        self.conv5_1 = conv3x3(512, 512)
        self.conv5_2 = conv3x3(512, 512)
        self.conv5_3 = conv3x3(512, 512)
        
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=0)
        self.dropout = nn.Dropout()
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0)
        
        # SEMANTIC SEGMENTAION PART
        self.scoring_layer = nn.Conv2d(4096, self.n_classes, kernel_size=1, 
                                      stride=1, padding=0)

        self.upscore2 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 
                                          kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(self.n_classes, self.n_classes,
                                         kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 
                                    kernel_size=16, stride=8, bias=False)
        
        # Initilize Weights
        self.scoring_layer.weight.data.zero_()
        self.scoring_layer.bias.data.zero_()
        
        self.score_pool3 = nn.Conv2d(256, self.n_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, self.n_classes, kernel_size=1)
        self.score_pool3.weight.data.zero_()
        self.score_pool3.bias.data.zero_()
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()

        self.upscore2.weight.data.copy_(get_upsampling_weight(self.n_classes, self.n_classes, 4))
        self.upscore_pool4.weight.data.copy_(get_upsampling_weight(self.n_classes, self.n_classes, 4))
        self.upscore8.weight.data.copy_(get_upsampling_weight(self.n_classes, self.n_classes, 16))

        # Pretrained layers
        pth_url = 'https://download.pytorch.org/models/vgg16-397923af.pth' # download from model zoo
        state_dict = model_zoo.load_url(pth_url)

        layer_names = [layer_name for layer_name in state_dict]

        
        counter = 0
        for p in self.parameters():
            if counter < 26:  # conv1_1 to pool5
                p.data = state_dict[ layer_names[counter] ]
            elif counter == 26:  # fc6 weight
                p.data = state_dict[ layer_names[counter] ].view(4096, 512, 7, 7)
            elif counter == 27:  # fc6 bias
                p.data = state_dict[ layer_names[counter] ]
            elif counter == 28:  # fc7 weight
                p.data = state_dict[ layer_names[counter] ].view(4096, 4096, 1, 1)
            elif counter == 29:  # fc7 bias
                p.data = state_dict[ layer_names[counter] ]


            counter += 1

    def forward(self, x):
        n,c,h,w = x.size()
        # VGG16 PART
        conv1_1 =  self.relu(  self.conv1_1(x) )
        conv1_2 =  self.relu(  self.conv1_2(conv1_1) )
        pool1 = self.pool(conv1_2)
        
        conv2_1 =  self.relu(   self.conv2_1(pool1) )
        conv2_2 =  self.relu(   self.conv2_2(conv2_1) )
        pool2 = self.pool(conv2_2)
        
        conv3_1 =  self.relu(   self.conv3_1(pool2) )
        conv3_2 =  self.relu(   self.conv3_2(conv3_1) )
        conv3_3 =  self.relu(   self.conv3_3(conv3_2) )
        pool3 = self.pool(conv3_3)
        
        conv4_1 =  self.relu(   self.conv4_1(pool3) )
        conv4_2 =  self.relu(   self.conv4_2(conv4_1) )
        conv4_3 =  self.relu(   self.conv4_3(conv4_2) )
        pool4 = self.pool(conv4_3)
        
        conv5_1 =  self.relu(   self.conv5_1(pool4) )
        conv5_2 =  self.relu(   self.conv5_2(conv5_1) )
        conv5_3 =  self.relu(   self.conv5_3(conv5_2) )
        pool5 = self.pool(conv5_3)
        
        fc6 = self.dropout( self.relu(   self.fc6(pool5) ) )
        fc7 = self.dropout( self.relu(   self.fc7(fc6) ) )
        
        # SEMANTIC SEGMENTATION PART
        # first
        scores = self.scoring_layer( fc7 )
        upscore2 = self.upscore2(scores)

        # second
        score_pool4 = self.score_pool4(pool4)
        score_pool4c = score_pool4[:, :, 5:5+upscore2.size(2), 
                                         5:5+upscore2.size(3)]
        upscore_pool4 = self.upscore_pool4(score_pool4c + upscore2)

        # third
        score_pool3 = self.score_pool3(pool3)
        score_pool3c = score_pool3[:, :, 9:9+upscore_pool4.size(2), 
                                         9:9+upscore_pool4.size(3)]

        output = self.upscore8(score_pool3c + upscore_pool4) 

        return output[:, :, 31: (31 + h), 31: (31 + w)].contiguous()

class FCN8_ResNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

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

# Utils
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), stride=(stride,stride),
                     padding=(padding,padding))

def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0)
