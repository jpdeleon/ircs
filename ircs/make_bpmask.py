#!/usr/bin/env python

# input: raw object frame list
# output: cal_flat.fits file in new FLAT directory (within input_dir)

import sys
from matplotlib import pyplot as plt
import numpy as np
try:
    from astropy.io import fits as pf
except:
    import pyfits as pf
from tqdm import tqdm

from ircs import utils

input_dir = '/mnt/sda1/data/ircs_pol/'
#input_dir = '/mnt/B838B30438B2C124/data/ircs_pol/'

def bp_mask(img, high, low, show_image, save_fits, cmap):
    '''
    creates a bad pixel mask given low and high threshhold
    '''
    if img is not None and img.ndim == 2:
        #check if img exists and is 2D
        pass
    else:
        #otherwise read from file
        img = pf.open(input_dir+'/calflat.fits')[0].data


    mask = np.copy(img)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if low <= mask[i,j] <= high:
                '''
                retain pixel value of flat or change to 1?
                '''
                mask[i,j] = 1 #good pixels
            else:
                mask[i,j] = 0 #bad pixels
    if show_image == True:
        plt.imshow(mask, cmap=cmap)
        plt.title('bad pixel mask')
        plt.colorbar()
        plt.show()
    if save_fits == True:
        try:
            #add header!
            pf.writeto(input_dir+'bp_mask.fits', mask)
            print('\nbp_mask.fits saved in {}\n'.format(input_dir))
        except:
            print('bp_mask.fits already exists!')
    return mask
