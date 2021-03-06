#!/usr/bin/env python

# input: raw object frame list
# output: cal_flat.fits file in new FLAT directory (within input_dir)

import sys
import os
from matplotlib import pyplot as plt
import numpy as np
try:
    from astropy.io import fits as pf
except:
    import pyfits as pf
from tqdm import tqdm

from ircs import utils

config = utils.check_config()
home_dir = config[0]
input_dir = config[1]
output_dir = config[2]

def bp_mask(img, high, low, show_image, save_fits, cmap):
    '''
    creates a bad pixel mask given low and high threshhold
    '''
    if img is not None and img.ndim == 2:
        #check if img exists and is 2D
        pass
    else:
        #otherwise read from file
        img = pf.open(os.path.join(input_dir,'calflat.fits'))[0].data


    mask = np.copy(img)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if low <= mask[i,j] <= high:
                mask[i,j] = True #good pixels
            else:
                mask[i,j] = False #bad pixels
    if show_image == True:
        plt.imshow(mask, cmap=cmap)
        plt.title('bad pixel mask')
        plt.colorbar()
        plt.show()
    if save_fits == True:
        try:
            #add header!
            pf.writeto(os.path.join(input_dir,'bp_mask.fits'), mask)
            print('\nbp_mask.fits saved in {}\n'.format(input_dir))
        except:
            print('bp_mask.fits already exists!')
    return mask
