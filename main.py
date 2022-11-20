import glob
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import logging

from model import Hydranet
from utils import plot_results, load_config

if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s - %(message)s')
    logging.getLogger().setLevel(logging.INFO)
    
    cfg = load_config('config/cfg.yaml')

    hydranet = Hydranet(cfg)

    images_files = glob.glob(cfg['general']['input_dir'] + cfg['general']['environment'] + '/*.png')
    idx = np.random.randint(0, len(images_files))

    img_path = images_files[idx]
    img = np.array(Image.open(img_path))

    mask, depth = hydranet.pipeline(img)

    plot_results(img, depth, mask, mask)
