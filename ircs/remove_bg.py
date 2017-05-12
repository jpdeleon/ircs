#!/usr/bin/env python

# input: obj list
# output: cropped fits files in new directory (within input_dir)

from glob import glob
import os
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
rmbg_input_dir = os.path.join(input_dir,'cropped')
rmbg_output_dir = os.path.join(rmbg_input_dir,'rm_bg')



file_list = glob(os.path.join(rmbg_input_dir,'*.fits'))
file_list.sort()

if os.listdir(rmbg_input_dir) != []:
    print('total no. of raw data frames: {0}\n'.format(len(file_list)))

def rm_bg(obj, box_size, show_before_after, save_fits, cmap):
    for i in tqdm(file_list):
        hdr=pf.open(i)[0].header
        image = np.copy(pf.open(i)[0].data)
        new_image = np.copy(pf.open(i)[0].data)
        for y in range(image.shape[0]):
            #take the median of y-values from 0:100
            #(101-499 is omitted because it contains bright star
            med1=np.median(image[y,:100])
            med2=np.median(image[y,-100:])
            #bg is subtracted by subtracting the average of the median
            #of pixels along x-values considered above to y-values
            new_image[y] = image[y] - (med1+med2)*0.5
        print('\nbackground level={}\n'.format((med1+med2)*0.5))
        if show_before_after == True:
            utils.compare_oe(image, new_image, hdr, hdr, cmap)
        if save_fits == True:
            try:
                fname=os.path.join(rmbg_output_dir,hdr['FRAMEID'],'_r.fits')
                pf.writeto(fname, new_image, hdr)
            except:
                print('{}.fits already exists!'.format(fname))
