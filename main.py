import glob
import logging

import numpy as np
from PIL import Image

from model import Hydranet
from utils import load_config, plot_results

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

    plot_results(img, depth, mask)
