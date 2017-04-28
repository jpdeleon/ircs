#!/usr/bin/env python

#CALFLAT
# input: flat_on and flat_off list
# output: cal_flat.fits file in new FLAT directory (within input_dir)

#SKYFLAT
# input: raw object frame list
# output: sky_flat.fits file in new FLAT directory (within input_dir)
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

'''
Don't forget the slash / in the end of input_dir!
'''
input_dir = '/mnt/sda1/data/ircs_pol/'
output_dir = input_dir+'flat/'
#input_dir = '/mnt/B838B30438B2C124/data/ircs_pol/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


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
            pf.writeto(input_dir+dark_name, master_flat_on_med)
            pf.writeto(input_dir+flat_name, calflat)
            print('\{0} and {1} saved in {2}!\n'.format(dark_name, flat_name, input_dir))
        except:
            print('calflat.fits already exists!')
    return master_flat_on_med

def skyflat(obj, bpmask, show_image, save_fits, cmap):
    '''
    creates a skyflat using raw frames and bad pixel mask from make_bpmask.py
    '''
    if bpmask is not None and bpmask.ndim == 2:
        #check if bpmask exists and is 2D
        pass
    else:
        #otherwise read from file
        bpmask = pf.open(input_dir+'bp_mask.fits')[0].data

    imgs_tmp = []
    imgs_norm = []

    #imgs_tmp = np.array([pf.getdata("%s" % n)/ np.mean(pf.getdata("%s" % n)) for n in obj])
    for i in obj:
        img= pf.open(i)[0].data
        img_norm = img / np.mean(img)
        imgs_tmp.append(img_norm)
    imgs_tmp_arr = np.asarray(imgs_tmp)
    '''
    np.median(,axis=0) gives nan; so nanmedian is used
    '''
    img_med = np.nanmedian(imgs_tmp_arr, axis= 0)

    for j in obj:
        img = pf.open(i)[0].data
        img /= img_med
        imgs_norm.append(img)

    imgs_norm_arr = np.asarray(imgs_norm)
    skyflat = np.nanmedian(imgs_norm_arr, axis=0)
    #skyflat /= bpmask

    if show_image == True:
        plt.imshow(skyflat, cmap=cmap)
        plt.title('sky flat')
        plt.colorbar()
        plt.show()
    if save_fits == True:
        try:
            #add header!
            pf.writeto(input_dir+'skyflat.fits', bpmask)
            print('\skyflat.fits saved in {}\n'.format(input_dir))
        except:
            print('skyflat.fits already exists!')
    return skyflat

def flat_div(obj, flat_name, show_image, save_fits, cmap):
    '''
    img_name can be calflat or skyflat
    '''

    if flat_name == 'calflat':
        #otherwise read from file
        flat = pf.open(input_dir+'calflat.fits')[0].data
    elif flat_name == 'skyflat':
        flat = pf.open(input_dir+'skyflat.fits')[0].data
    else:
        #ask and run again
        flat_name = raw_input("Choose: ['calflat','skyflat']:\n")
        flat_div(flat_name, show_image, save_fits, cmap)
    for i in obj:
        flattened_img= pf.open(i)[0].data
        hdr = pf.open(i)[0].header
        flattened_img /= flat

        try:
            '''
            bug: ENTER causes every succeeding image to not show but save
            '''
            response #none or ENTER
            #means quit is pressed before so skip showing image
            pass
        except:
            if show_image == True:
                plt.ion()
                plt.imshow(flattened_img, cmap=cmap)
                plt.title('Flattened OBJ frames')
                plt.colorbar()
                plt.show()

                if sys.version_info>(2,7,14): #check if newer than python 2.7.13
                    response = input("Press [enter] to continue; [q] to quit display...")
                else: #use python 2
                    response = raw_input("Press [enter] to continue; [q] to quit display...")
                '''
                ask this first time, if q, image will not be shown and images wil still be saved
                '''
                if response.lower in ['q', 'quit', 'qu', 'exit']:
                    break # quit display
                else:
                    pass
                plt.close()

        if save_fits == True:
            try:
                #add header!
                fname=hdr['FRAMEID']+'_f.fits'
                pf.writeto(output_dir+fname, flattened_img)
                print('\n{0} saved in {1}\n'.format(fname, output_dir))
            except:
                print('{} already exists!'.format(hdr['FRAMEID']))
    return None
