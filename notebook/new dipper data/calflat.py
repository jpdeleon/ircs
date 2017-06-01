#!/usr/bin/env python

from glob import glob
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
try:
    from astropy.io import fits as pf
    from astropy.stats import sigma_clip
except:
    import pyfits as pf
from tqdm import tqdm
import getpass
import utils


params = 'FRAMEID, DATA-TYP, OBJECT, EXP1TIME, COADD, D_MODE, I_SCALE, \
                I_DTHNUM, I_DTHPOS'
file_list = glob('IRCA*.fits')
file_list.sort()

input_dir = os.path.join('/home',getpass.getuser(),'data/ircs_UH30B/CALIB/FLAT')
flat_output_dir = input_dir

if os.listdir(input_dir) != []:
        print('total no. of raw data frames: {0}\n'.format(len(file_list)))

summary = []
flat=[]
flat_type=[]
flat_off=[]
flat_on=[]
others=[]

for i in tqdm(file_list):
    hdr=pf.open(i)[0].header
    #get each params in header
    summary.append([hdr[j] for j in params.split(',')])
    if hdr['DATA-TYP'] ==  'FLAT':
    	flat.append(i)
    	flat_type.append(hdr['OBJECT'])
    	if 'OFF' in hdr['OBJECT'].split()[0].split('_'):
            #'IMAGE_Kp_OFF HWP0'
            #'IMAGE_Kp_OFF HWP22.5'
            #'IMAGE_Kp_OFF HWP45'
            #'IMAGE_Kp_OFF HWP67.5'
            flat_off.append(i)
    	else:
            flat_on.append(i)
    else: #hdr['DATA-TYP'] ==  'DARK'?
        others.append(i)
np.savetxt(os.path.join(input_dir,'flat_off.txt'), flat_off, fmt="%s", delimiter=',')
np.savetxt(os.path.join(input_dir,'flat_on.txt'), flat_on, fmt="%s", delimiter=',')
np.savetxt(os.path.join(input_dir,'others.txt'), others, fmt="%s", delimiter=',')
print('OBJECT and FLAT lists saved in {}\n'.format(input_dir))
print('\nSee also summary.txt\n')


try:
    flat_off=np.loadtxt('flat_off.txt', dtype=str, delimiter=',')
    flat_on=np.loadtxt('flat_on.txt', dtype=str, delimiter=',')
    #others=np.loadtxt(os.path.join(data_dir,'others.txt'), dtype=str, delimiter=',')
except:
    print('Missing text files!\nHave you run image-sorter?\n')

if not os.path.exists(flat_output_dir):
    os.makedirs(flat_output_dir)

def calflat(on, off, check_frames, show_image, save_fits, cmap):
    '''
    corrects for sensitivity differences over the detector

    1. ON-OFF frame = each ON frame - median(OFF frames)
    2. (each ON- OFF frame)/mean(ON-OFF)
    3. combine result of step 2 with meadian
    '''
    print('Creating flat...\n')
    flats_on = np.array([pf.getdata("%s" % n) for n in on])
    flats_off = np.array([pf.getdata("%s" % m) for m in off])

    if check_frames == True:
        print('Checking FLAT_ON')
        utils.check_frame(on, calc='subtract', show_image=True, cmap=cmap)
        print('Checking FLAT_OFF')
        utils.check_frame(off, calc='subtract', show_image=True, cmap=cmap)
    print('(If you did not find bad frames)')
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

    flat_name = 'calflat.fits'
    dark_name = 'master_flat_off.fits'
    if show_image == True:
        plt.figure(1)
        plt.imshow(master_flat_on_med, cmap=cmap)
        plt.title(flat_name)
        plt.figure(2)
        n, bins, patches = plt.hist(master_flat_on_med.ravel(), bins=256, alpha=0.75)
        plt.title('Histogram of {}'.format(flat_name))
        plt.figure(3)
        plt.imshow(master_flat_off, cmap=cmap)
        plt.title(dark_name)
        plt.figure(4)
        n, bins, patches = plt.hist(master_flat_off.ravel(), bins=256, alpha=0.75)
        plt.title('Histogram of {}'.format(dark_name))
        plt.show()
    if save_fits == True:
        try:
            #add header!
            pf.writeto(os.path.join(input_dir,dark_name), master_flat_off)
            pf.writeto(os.path.join(input_dir,flat_name), master_flat_on_med)
            print('\{0} and {1} saved in {2}!\n'.format(dark_name, flat_name, input_dir))
        except:
            print('calflat.fits already exists!')
    return master_flat_on_med


calflat_img = calflat(flat_on, flat_off,
                            check_frames=False,
                            show_image=False,
                            save_fits=True,
                            cmap='gray')

