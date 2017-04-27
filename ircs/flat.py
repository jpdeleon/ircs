#!/usr/bin/env python

# input: flat_on and flat_off list
# output: cal_flat.fits file in new FLAT directory (within input_dir)
from matplotlib import pyplot as plt
import numpy as np
try:
    from astropy.io import fits as pf
except:
    import pyfits as pf
from tqdm import tqdm

from ircs import utils

def calflat(on, off, check_frames, show_image, cmap, save_fits):
    '''
    corrects for sensitivity differences over the detector
    '''
    print('Creating flat...\n')
    flats_on = np.array([pf.getdata("%s" % n) for n in on])
    flats_off = np.array([pf.getdata("%s" % m) for m in off])
    import pdb; pdb.set_trace()
    if check_frames == True:
        print('Checking FLAT_ON')
        utils.check_frame(on, calc='subtract', show_image=check_frames, cmap)
        print('Checking FLAT_OFF')
        utils.check_frame(off, calc='subtract', show_image=check_frames, cmap)
    print('Did you find any bad frames?')
    if utils.proceed():
        pass
    else:
        print('\n-----------------------------------------\n')
        print('Delete the entry in flat_on/flat_off.txt.\nThen, run this script again.')
        print('-----------------------------------------\n')
        sys.exit()

    master_flat_off = np.median(flats_off,axis=0)
    for i in len(flats_on):
        master_flat_on = (flats_on[i] - master_flat_off)/np.mean(flats_on[i])
    #master_flat_on = np.array()
    master_flat_on_normed = np.median(master_flat_on, axis=0)
    if show_image == True:
        plt.show(master_flat_on_normed, cmap=cmap)
    if save_fits == True:
        try:
            #add header!
            pf.writeto('calflat.fits', master_flat_on_normed)
        except:
            print('calflat.fits already exists!')
    return None
