import matplotlib.pyplot as plt
import numpy as np


def plot_results(img, depth, mask):
    # overlay segmentation on top of the image
    segmented = (mask * 0.5 + img * 0.5).astype(np.uint8)

    #plot original image over the other three with subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharey=True, sharex=True)
    for a in axs:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_aspect('equal')
    fig.subplots_adjust(wspace=0, hspace=0)
    axs[0].imshow(img)
    axs[1].imshow(depth, cmap='plasma')
    axs[2].imshow(mask)
    axs[3].imshow(segmented)

    plt.show()
