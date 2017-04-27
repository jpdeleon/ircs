#!/usr/bin/env python

# input: flat_on and flat_off list
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

'''
Don't forget the slash / in the end of input_dir!
'''
input_dir = '/mnt/sda1/data/ircs_pol/'
#input_dir = '/mnt/B838B30438B2C124/data/ircs_pol/'

def calflat(on, off, check_frames, show_image, save_fits, cmap):
    '''
    corrects for sensitivity differences over the detector
    '''
    print('Creating flat...\n')
    flats_on = np.array([pf.getdata("%s" % n) for n in on])
    flats_off = np.array([pf.getdata("%s" % m) for m in off])

    if check_frames == True:
        print('Checking FLAT_ON')
        utils.check_frame(on, calc='subtract', show_image=False, cmap=cmap)
        print('Checking FLAT_OFF')
        utils.check_frame(off, calc='subtract', show_image=False, cmap=cmap)
    print('Continue (if you did not find bad frames) [y/n]?')
    if utils.proceed():
        pass
    else:
        print('\n-----------------------------------------\n')
        print('Delete the entry in flat_on/flat_off.txt.\nThen, run this script again.')
        print('-----------------------------------------\n')
        sys.exit()

    master_flat_off = np.median(flats_off,axis=0)
    master_flat_on = []
    for i in range(len(flats_on)):
        master_flat_on.append((flats_on[i] - master_flat_off)/np.mean(flats_on[i]))

    master_flat_on_med = np.median(master_flat_on, axis=0)

    if show_image == True:
        plt.imshow(master_flat_on_med, cmap=cmap)
        plt.title('calflat.fits')
        plt.show()
    if save_fits == True:
        try:
            #add header!
            pf.writeto(input_dir+'calflat.fits', master_flat_on_med)
            print('\ncalflat.fits saved in {}!\n'.format(input_dir))
        except:
            print('calflat.fits already exists!')
    return None
