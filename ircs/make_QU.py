#!/usr/bin/env python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
try:
    from astropy.io import fits as pf
except:
    import pyfits as pf
from tqdm import tqdm

from glob import glob
from ircs import utils

config = utils.check_config()
home_dir = config[0]
input_dir = config[1]
oe_dir = config[6]

file_list_o = glob(os.path.join(oe_dir,'*_o.fits'))
file_list_e = glob(os.path.join(oe_dir,'*_e.fits'))
file_list_o.sort()
file_list_e.sort()


if not os.path.exists(oe_dir):
    os.makedirs(oe_dir)
    if not os.path.exists(oe_dir):
        os.makedirs(oe_dir)

def QU(image_o, image_e, show_QU=True, save_fits=False, cmap='gray'):
    '''
    Q=WP0-WP45
    U=WP22.5-67.5
    '''
    if image_o and image_e is not None: #and image_o.ndim ==2 and image_e.ndim == 2:
        #check if img exists and is 2D
        pass
    else:
        image_o = {}
        image_e = {}
        #otherwise read from file
        if len(file_list_o)>0:
            for i in file_list_o:
                dither = pf.open(i)[0].header['I_DTHPOS']
                dither_position = dither.split(':')[0].strip()
                image = np.copy(pf.open(i)[0].data)
                image_o[dither]=image

            for j in file_list_e:
                dither = pf.open(j)[0].header['I_DTHPOS']
                dither_position = dither.split(':')[0].strip()
                image = np.copy(pf.open(j)[0].data)
                image_e[dither]=image
        else:
            print('\n{} seems empty.'.format(oe_dir))
            #EXITING
            sys.exit()

    for k in tqdm(np.arange(1,6,1)):#header_o[dither]['I_DTHNUM']):
        Q = image_o[str(k)+' : WP0'] - image_o[str(k)+' : WP45']
        U = image_o[str(k)+' : WP22.5'] - image_o[str(k)+' : WP67.5']
        I = image_o[str(k)+' : WP0'] + image_o[str(k)+' : WP45']
        vmin1, vmax1 = None, None#np.median(Q), 10*np.median(Q)
        vmin2, vmax2 = None, None#np.median(U), 10*np.median(U)

        if show_QU == True:
            plt.ion()

            fig,ax = plt.subplots(nrows=1,ncols=2, figsize=(10,5))
            ax1 = ax[0].imshow(Q,cmap=cmap,vmin=vmin1,vmax=vmax1)
            ax[0].set_title('Q= WP0-WP45')
            ax[0].set_xlabel('dither={}'.format(k))
            fig.colorbar(ax1, ax=ax[0])

            ax2 = ax[1].imshow(U,cmap=cmap,vmin=vmin2,vmax=vmax2)
            ax[1].set_title('U= WP22.5-WP67.5')
            ax[1].set_xlabel('dither={}'.format(k))
            fig.colorbar(ax2, ax=ax[1])
            plt.show()

            fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(5,5))

            #try/except still allows both to run
            if sys.version_info>(2,7,14): #check if newer than python 2.7.13
                response = input("Press [enter] to continue...")
            else: #use python 2
                response = raw_input("Press [enter] to continue...")
            plt.close()


        if save_fits == True:
            try:
                pf.writeto(os.path.join(oe_dir,header_o[dither]['FRAMEID'],'_o.fits'), image_o[dither], header_o[dither])
            except:
                print('{}_o.fits already exists!'.format(header_o[dither]['FRAMEID']))
            try:
                pf.writeto(os.path.join(oe_dir,header_e[dither]['FRAMEID'],'_e.fits'), image_e[dither], header_e[dither])
            except:
                print('{}_e.fits already exists!'.format(header_o[dither]['FRAMEID']))
