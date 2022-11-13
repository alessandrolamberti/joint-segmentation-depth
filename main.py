import glob
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable

from model import Hydranet
from utils import plot_results, preprocess_image

if __name__ == '__main__':
    hydranet = Hydranet()

    CMAP = np.load('data/hydranets-data/hydranets-data/cmap_kitti.npy')
    NUM_CLASSES = 6

    images_files = glob.glob('data/residential/*.png')
    idx = np.random.randint(0, len(images_files))

    img_path = images_files[idx]
    img = np.array(Image.open(img_path))

    def inference(img):
            img_var = preprocess_image(img, cuda=True)
            segm, depth = hydranet.pipeline(img_var)
            segm = cv2.resize(segm[0, :NUM_CLASSES].cpu().data.numpy().transpose(1, 2, 0),
                            img.shape[:2][::-1],
                            interpolation=cv2.INTER_CUBIC)
            depth = cv2.resize(depth[0, 0].cpu().data.numpy(),
                            img.shape[:2][::-1],
                            interpolation=cv2.INTER_CUBIC)
            segm = CMAP[segm.argmax(axis=2)].astype(np.uint8)
            depth = np.abs(depth)
            return depth, segm

    depth, mask = inference(img)
    plot_results(img, depth, mask, mask)
