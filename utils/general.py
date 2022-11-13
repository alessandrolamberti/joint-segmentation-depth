from torch.autograd import Variable
import torch
import numpy as np

def preprocess_image(image, cuda):
    IMG_SCALE  = 1./255
    IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    img = (image * IMG_SCALE - IMG_MEAN) / IMG_STD
    img_var = Variable(torch.from_numpy(img.transpose(2, 0, 1)[None]), requires_grad=False).float()
    if cuda:
        img_var = img_var.cuda()
    return img_var