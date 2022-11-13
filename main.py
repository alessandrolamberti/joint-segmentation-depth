import glob
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from model import Hydranet
from utils import plot_results, preprocess_image, postprocess_images

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
            mask, depth = hydranet.pipeline(img_var)
            mask, depth = postprocess_images((img, mask, depth), NUM_CLASSES, CMAP)
            return mask, depth

    mask, depth = inference(img)
    plot_results(img, depth, mask, mask)
