import logging

import cv2
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from model.utils import (CRPBlock, InvertedResidualBlock, batchnorm, conv1x1,
                         conv3x3, convbnrelu)
from utils.general import timeit


class Hydranet(nn.Module):

    def __init__(self, cfg):        
        super().__init__()
        self.cfg = cfg
        self.num_classes = self.cfg['model']['num_classes']

        self.define_mobilenet() # define the MobileNet backbone
        self.define_lightweight_refinenet() # define the Light-Weight RefineNet
        self.load_weights(path=self.cfg['model']['weights'])
        self.eval().to(self.cfg['model']['device'])
        self.warmup()
        logging.info('Model loaded on device: {}'.format(self.cfg['model']['device']))

    def define_mobilenet(self):
        mobilenet_config = self.cfg['model']['encoder_config']
        self.in_channels = 32 # input channels
        self.layer1 = convbnrelu(3, self.in_channels, kernel_size=3, stride=2)
        c_layer = 2
        for t,c,n,s in (mobilenet_config):
            layers = []
            for idx in range(n):
                layers.append(InvertedResidualBlock(self.in_channels, c, expansion_factor=t, stride=s if idx == 0 else 1))
                self.in_channels = c
            setattr(self, 'layer{}'.format(c_layer), nn.Sequential(*layers))
            c_layer += 1

    def make_crp(self, in_planes, out_planes, stages, groups=False):
        layers = [CRPBlock(in_planes, out_planes,stages, groups=groups)]
        return nn.Sequential(*layers)

    def define_lightweight_refinenet(self):
        ## Light-Weight RefineNet ##
        self.conv8 = conv1x1(320, 256, bias=False)
        self.conv7 = conv1x1(160, 256, bias=False)
        self.conv6 = conv1x1(96, 256, bias=False)
        self.conv5 = conv1x1(64, 256, bias=False)
        self.conv4 = conv1x1(32, 256, bias=False)
        self.conv3 = conv1x1(24, 256, bias=False)
        self.crp4 = self.make_crp(256, 256, 4, groups=False)
        self.crp3 = self.make_crp(256, 256, 4, groups=False)
        self.crp2 = self.make_crp(256, 256, 4, groups=False)
        self.crp1 = self.make_crp(256, 256, 4, groups=True)

        self.conv_adapt4 = conv1x1(256, 256, bias=False)
        self.conv_adapt3 = conv1x1(256, 256, bias=False)
        self.conv_adapt2 = conv1x1(256, 256, bias=False)

        self.pre_depth = conv1x1(256, 256, groups=256, bias=False)
        self.depth = conv3x3(256, 1, bias=True)
        self.pre_segm = conv1x1(256, 256, groups=256, bias=False)
        self.segm = conv3x3(256, self.num_classes, bias=True)
        self.relu = nn.ReLU6(inplace=True)

    def warmup(self):
        if torch.cuda.is_available() and self.cfg['model']['device'] == 'cuda':
            img = torch.randn(1, 3, 224, 224).cuda()
            self.forward(img)

    def forward(self, x):
        # MOBILENET V2
        x = self.layer1(x)
        x = self.layer2(x) # x / 2
        l3 = self.layer3(x) # 24, x / 4
        l4 = self.layer4(l3) # 32, x / 8
        l5 = self.layer5(l4) # 64, x / 16
        l6 = self.layer6(l5) # 96, x / 16
        l7 = self.layer7(l6) # 160, x / 32
        l8 = self.layer8(l7) # 320, x / 32

        # LIGHT-WEIGHT REFINENET
        l8 = self.conv8(l8)
        l7 = self.conv7(l7)
        l7 = self.relu(l8 + l7)
        l7 = self.crp4(l7)
        l7 = self.conv_adapt4(l7)
        l7 = nn.Upsample(size=l6.size()[2:], mode='bilinear', align_corners=False)(l7)

        l6 = self.conv6(l6)
        l5 = self.conv5(l5)
        l5 = self.relu(l5 + l6 + l7)
        l5 = self.crp3(l5)
        l5 = self.conv_adapt3(l5)
        l5 = nn.Upsample(size=l4.size()[2:], mode='bilinear', align_corners=False)(l5)

        l4 = self.conv4(l4)
        l4 = self.relu(l5 + l4)
        l4 = self.crp2(l4)
        l4 = self.conv_adapt2(l4)
        l4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=False)(l4)

        l3 = self.conv3(l3)
        l3 = self.relu(l3 + l4)
        l3 = self.crp1(l3)

        # HEADS
        out_segm = self.pre_segm(l3)
        out_segm = self.relu(out_segm)
        out_segm = self.segm(out_segm)

        out_d = self.pre_depth(l3)
        out_d = self.relu(out_d)
        out_d = self.depth(out_d)

        return out_segm, out_d

    def load_weights(self, path):
        ckpt = torch.load(path)
        self.load_state_dict(ckpt['state_dict'])

    def preprocess_image(self, image):
        IMG_MEAN = np.array(self.cfg['preprocessing']['img_mean']).reshape((1, 1, 3))
        IMG_STD = np.array(self.cfg['preprocessing']['img_std']).reshape((1, 1, 3))

        img = (image * 1./255 - IMG_MEAN) / IMG_STD
        img_var = Variable(torch.from_numpy(img.transpose(2, 0, 1)[None]), requires_grad=False).float()
        if torch.cuda.is_available() and self.cfg['model']['device'] == 'cuda':
            img_var = img_var.cuda()
        return img_var

    def postprocess_images(self, orig, segm, depth):
        segm = cv2.resize(segm[0, :self.num_classes].cpu().data.numpy().transpose(1, 2, 0),
                        orig.shape[:2][::-1],
                        interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth[0, 0].cpu().data.numpy(),
                        orig.shape[:2][::-1],
                        interpolation=cv2.INTER_CUBIC)

        cmap = np.load(self.cfg['general']['cmap'])
        segm = cmap[segm.argmax(axis=2)].astype(np.uint8)
        depth = np.abs(depth)
        return segm, depth

    @timeit
    def pipeline(self, image):
        img_var = self.preprocess_image(image)
        with torch.no_grad():
            mask, depth = self.forward(img_var)

        return self.postprocess_images(image, mask, depth)



    
