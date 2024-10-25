import numpy as np
import os

# Directory Check
def check_dir(path):
    """
    Check the directory or Make the directory
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print('Error: Failed to create the directory.')

# 개선된 가우시안 컨볼루션
def pseudo_conv2d(image, sigma=1, padding=2, scaling=1):
    image_pad = np.pad(image, ((padding,)*2,)*2, 'constant', constant_values=0)
    feature = np.zeros(image_pad.shape)
    idx = np.nonzero(image_pad)
    for x in range(image_pad.shape[0]):
        for y in range(image_pad.shape[1]):
            for i in range(len(idx[0])):
                feature[x,y] += np.exp(-((x-idx[0][i])**2+(y-idx[1][i])**2)/(2*(sigma**2)))/(2*np.pi*(sigma**2))
    feature *= scaling
    return feature