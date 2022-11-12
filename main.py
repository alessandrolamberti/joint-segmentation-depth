import glob
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable

from model import Hydranet

if __name__ == '__main__':
    print(torch.version.cuda)
    hydranet = Hydranet()
    print(hydranet)
    ckpt = torch.load('data/hydranets-data/hydranets-data/ExpKITTI_joint.ckpt')
    hydranet.load_state_dict(ckpt['state_dict'])
    hydranet.eval()
    if torch.cuda.is_available():
        hydranet.cuda()

    IMG_SCALE  = 1./255
    IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    def prepare_img(img):
        return (img * IMG_SCALE - IMG_MEAN) / IMG_STD

    CMAP = np.load('data/hydranets-data/hydranets-data/cmap_kitti.npy')
    NUM_CLASSES = 6

    images_files = glob.glob('data/hydranets-data/hydranets-data/data/*.png')
    idx = np.random.randint(0, len(images_files))

    img_path = images_files[idx]
    img = np.array(Image.open(img_path))

    def pipeline(img):
        with torch.no_grad():
            img_var = Variable(torch.from_numpy(prepare_img(img).transpose(2, 0, 1)[None]), requires_grad=False).float()
            if torch.cuda.is_available():
                img_var = img_var.cuda()
            segm, depth = hydranet(img_var)
            segm = cv2.resize(segm[0, :NUM_CLASSES].cpu().data.numpy().transpose(1, 2, 0),
                            img.shape[:2][::-1],
                            interpolation=cv2.INTER_CUBIC)
            depth = cv2.resize(depth[0, 0].cpu().data.numpy(),
                            img.shape[:2][::-1],
                            interpolation=cv2.INTER_CUBIC)
            segm = CMAP[segm.argmax(axis=2)].astype(np.uint8)
            depth = np.abs(depth)
            return depth, segm

    depth, segm = pipeline(img)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30,20))
    ax1.imshow(img)
    ax1.set_title('Original', fontsize=30)
    ax2.imshow(segm)
    ax2.set_title('Predicted Segmentation', fontsize=30)
    ax3.imshow(depth, cmap="plasma", vmin=0, vmax=80)
    ax3.set_title("Predicted Depth", fontsize=30)
    plt.show()
