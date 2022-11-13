from torch.autograd import Variable
import torch
import numpy as np
import cv2

def preprocess_image(image, cuda):
    IMG_SCALE  = 1./255
    IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    img = (image * IMG_SCALE - IMG_MEAN) / IMG_STD
    img_var = Variable(torch.from_numpy(img.transpose(2, 0, 1)[None]), requires_grad=False).float()
    if cuda:
        img_var = img_var.cuda()
    return img_var

def postprocess_images(images, nc, cmap):
    img, segm, depth = images
    segm = cv2.resize(segm[0, :nc].cpu().data.numpy().transpose(1, 2, 0),
                    img.shape[:2][::-1],
                    interpolation=cv2.INTER_CUBIC)
    depth = cv2.resize(depth[0, 0].cpu().data.numpy(),
                    img.shape[:2][::-1],
                    interpolation=cv2.INTER_CUBIC)
    segm = cmap[segm.argmax(axis=2)].astype(np.uint8)
    depth = np.abs(depth)
    return segm, depth
