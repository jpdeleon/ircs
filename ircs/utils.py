#!/usr/bin/env python
from glob import glob
import os
import sys
import numpy as np
try:
    from astropy.io import fits as pf
except:
    import pyfits as pf
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from photutils.centroids import centroid_com as com
from photutils import CircularAperture
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
import pandas as pd

#input_dir = '/home/jp/data/ircs_pol'
input_dir = '/mnt/sda1/data/ircs_pol'
#input_dir = '/mnt/B838B30438B2C124/data/ircs_pol'
output_dir = '/home/Jerome/ircs_pol_output'

ircs_pix_size = 20.57*1e-3
strip_width = 4.4
separation = strip_width/ircs_pix_size
## Data Reduction Pipeline

def proceed():
    '''
    prompts the user to continue or stop in between steps in data reduction
    '''
    yes = set(['yes','y', 'ye', ''])
    no = set(['no','n'])

    choice = raw_input('Proceed to next step? [yes/no]\n').lower()
    if choice in yes:
       return True
    elif choice in no:
       return False
    else:
       sys.stdout.write("Please respond with 'yes' or 'no'.\n")
       proceed()

def check_header(fname):
    '''
    simple header printer in terminal
    similar to: from astropy.io.fits import getheader
    '''
    sample_hdr = pf.open(os.path.join(input_dir,fname))[0].header
    print(sample_hdr)
    return sample_hdr

def image_sorter(input_dir, save_list=True):
    '''
    sort images inside input_dir based on header['OBJECT']
    input_dir can be changed if needed
    '''
    file_list = glob(os.path.join(input_dir,'*.fits'))
    file_list.sort()

    if os.listdir(input_dir) != []:
        print('total no. of raw data frames: {0}\n'.format(len(file_list)))
    summary = []
    obj=[]
    obj_type=[]
    flat=[]
    flat_type=[]
    flat_off=[]
    flat_on=[]
    others=[]
    #parameters to extract from header
    params = 'FRAMEID, DATA-TYP, OBJECT, EXP1TIME, COADD, D_MODE, I_SCALE, I_DTHNUM, I_DTHPOS'
    for i in tqdm(file_list):
        hdr=pf.open(i)[0].header
        #get each params in header
        summary.append([hdr[j] for j in params.split(',')])
        if hdr['DATA-TYP'] ==  'OBJECT':
            obj.append(i)
            obj_type.append(hdr['OBJECT'])
        elif hdr['DATA-TYP'] ==  'FLAT':
            flat.append(i)
            flat_type.append(hdr['OBJECT'])
            if hdr['OBJECT'].split()[0].split('_')[2] == 'OFF':
                #'IMAGE_Kp_OFF HWP0'
                #'IMAGE_Kp_OFF HWP22.5'
                #'IMAGE_Kp_OFF HWP45'
                #'IMAGE_Kp_OFF HWP67.5'
                flat_off.append(i)
            else:
                flat_on.append(i)
        else: #hdr['DATA-TYP'] ==  'DARK'?
            others.append(i)

    print('\nOBJECT:\n{}\n'.format(set(obj_type)))

    #save header summary by default
    np.savetxt(os.path.join(input_dir,'summary.txt'), summary, header=params, fmt="%s", delimiter=',')

    if len(set(obj_type))>1:
        print('\n=====================WARNING=====================\n')
        print('\n{0} objects found in\n{1}'.format(len(set(obj_type)), input_dir))
        print('It is a MUST to have only ONE object per directory\n')
        print('\nMOVE the other OBJECT in a separate directory and run this again')
        print('\nSee summary.txt\n')
        print('\n=====================WARNING=====================\n')
        #EXITING
        sys.exit()
    #import pdb; pdb.set_trace()

    print('Types of FLAT:\n{}\n'.format(set(flat_type)))
    #save into text file
    if save_list == True:
        np.savetxt(os.path.join(input_dir,'object_types.txt'), list(set(obj_type)), fmt="%s", delimiter=',')
        np.savetxt(os.path.join(input_dir,'object.txt'), obj, fmt="%s", delimiter=',')
        np.savetxt(os.path.join(input_dir,'flat_off.txt'), flat_off, fmt="%s", delimiter=',')
        np.savetxt(os.path.join(input_dir,'flat_on.txt'), flat_on, fmt="%s", delimiter=',')
        np.savetxt(os.path.join(input_dir,'others.txt'), others, fmt="%s", delimiter=',')
        print('OBJECT and FLAT lists saved in {}\n'.format(input_dir))
        print('\nSee also summary.txt\n')

    return obj, flat_on, flat_off, others

def check_frame(lst, calc, show_image, cmap):
    '''
    calculates difference or dividend between two consecutive frames and displays it
    '''
    for idx, i in tqdm(enumerate(lst[:-2])):
        hdr1 = pf.open(i)[0].header
        img1 = pf.open(i)[0].data
        hdr2 = pf.open(lst[idx+1])[0].header
        img2 = pf.open(lst[idx+1])[0].data
        bkg_1 = np.median(img1)/hdr1['NDR']/hdr1['COADD']
        bkg_2 = np.median(img1)/hdr2['NDR']/hdr2['COADD']
        plt.ion()
        if calc == 'divide':
            if show_image == True:
                plt.imshow(img1 / img2, cmap=cmap)
                plt.title('{0} / {1}'.format(hdr1['FRAMEID'], hdr2['FRAMEID']))
                plt.xlabel('({0}) ; ({1})'.format(hdr1['I_DTHPOS'], hdr2['I_DTHPOS']))
                plt.colorbar()
            print('\nbackground levels:\n{0}: {1}\n{2}: {3}\n'.format(hdr1['FRAMEID'], bkg_1, hdr2['FRAMEID'], bkg_2))

        else: # calc == 'subtract':
            if show_image == True:
                plt.imshow(img1 - img2, cmap=cmap)
                plt.title('{0} - {1}'.format(hdr1['FRAMEID'], hdr2['FRAMEID']))
                plt.xlabel('{0} : {1}'.format(hdr1['I_DTHPOS'], hdr2['I_DTHPOS']))
                plt.colorbar()
            print('\nbackground levels:\n{0}: {1}\n{2}: {3}\n'.format(hdr1['FRAMEID'], bkg_1, hdr2['FRAMEID'], bkg_2))
        plt.show()
        if sys.version_info>(2,7,14): #check if newer than python 2.7.13
            response = input("Press [enter] to continue; [q] to quit display...")
        else: #use python 2
            response = raw_input("Press [enter] to continue; [q] to quit display...")

        if response.lower in ['q', 'quit', 'qu', 'exit']:
            break # quit display

        plt.close()

def show_hist(image, bins):
    n, bins, patches = plt.hist(image.ravel(), bins=256, alpha=0.75)

def test_image(pol, unpol, on, off):
    '''
    displays two images side-by-side;
    currently useless
    '''

    pol_image = pf.open(pol[0])[0].data
    unpol_image = pf.open(unpol[0])[0].data

    # on_image = pf.open(on[0])[0].data
    # off_image = pf.open(off[0])[0].data

    vmin1, vmax1 = np.median(pol_image), 10*np.median(pol_image)
    vmin2, vmax2 = np.median(unpol_image), 10*np.median(unpol_image)

    fig,ax = plt.subplots(nrows=1,ncols=2, figsize=(10,5))
    ax1 = ax[0].imshow(pol_image,vmin=vmin1,vmax=vmax1)
    ax[0].set_title(pf.open(pol[0])[0].header['OBJECT'])
    ax[0].set_xlabel(pol[0].split("/")[-1])
    fig.colorbar(ax1, ax=ax[0])

    ax2 = ax[1].imshow(unpol_image,vmin=vmin2,vmax=vmax2)
    ax[1].set_title(pf.open(unpol[0])[0].header['OBJECT'])
    ax[1].set_xlabel(unpol[0].split("/")[-1])
    fig.colorbar(ax2, ax=ax[1])

    #plt.imshow(unpol_image)
    #plt.imshow(pol_image)
    #plt.imshow(pol_image)
    plt.show()

def compare_oe(pol_image, unpol_image, header_o, header_e, cmap):
    '''
    compares o- and e-ray and displays them side-by-side

    Due to remove_bg, the np.median(image) produces higher bkg_o
    than original raw image
    '''
    box_size = 150
    if cmap is None:
        cmap=None

    elif cmap == 'gray':
        cmap='gray'

    else: #if None
        cmap='jet'
    pol_name = header_o['OBJECT']
    unpol_name = header_e['OBJECT']

    print('\nShowing {0} and {1}...\n'.format(pol_name,unpol_name))
    dither_step=header_o['I_DTHSZ']
    print('dither step={0} = {1} pix\n'.format(dither_step, dither_step/ircs_pix_size))
    # turn on interactive mode, non-blocking `show`
    plt.ion()
    #import pdb; pdb.set_trace()
    vmin1, vmax1 = None, None#np.median(pol_image), 10*np.median(pol_image)
    vmin2, vmax2 = None, None#np.median(unpol_image), 10*np.median(unpol_image)

    fig,ax = plt.subplots(nrows=1,ncols=2, figsize=(10,5))
    ax1 = ax[0].imshow(pol_image,cmap=cmap,vmin=vmin1,vmax=vmax1)
    #ax[0].set_title(header_o['OBJECT'])
    ax[0].set_title('vertically polarized')
    peak_flux_o, bkg_o = get_peak_flux(pol_image, header_o, box_size, r=3)
    ax[0].set_xlabel('peak flux/NDR/coadd\n={0}\nbackground={1}\n'.format(peak_flux_o, bkg_o))
    fig.colorbar(ax1, ax=ax[0])

    ax2 = ax[1].imshow(unpol_image,cmap=cmap,vmin=vmin2,vmax=vmax2)
    #ax[1].set_title(header_e['OBJECT'])
    ax[1].set_title('horizontally polarized')
    peak_flux_e, bkg_e = get_peak_flux(unpol_image, header_e, box_size, r=3)
    ax[1].set_xlabel('peak flux/NDR/coadd\n={0}\nbackground={1}\n'.format(peak_flux_e, bkg_e))
    fig.colorbar(ax2, ax=ax[1])
    plt.suptitle('{0}, {1}'.format(header_o['OBJECT'], header_e['FRAMEID']))
    plt.show()
    print('Dither number, waveplate position = {}\n'.format(header_e['I_DTHPOS']))
    '''
    bug: input works but shows EOF error!
    raw_input works but might not work for python 3!
    results to twice ENTER to proceed
    '''
    #try/except still allows both to run
    if sys.version_info>(2,7,14): #check if newer than python 2.7.13
        response = input("Press [enter] to continue...")
    else: #use python 2
        response = raw_input("Press [enter] to continue...")

    # if response == '':
    #     pass
    # elif response.lower() in ['q','quit','exit']:
    #     return False #quit
    plt.close()

def get_peak_flux(image, header, box_size, r):
    '''try:
            _ = input("Press [enter] to continue...")
        except:
            _ = raw_input("Press [enter] to continue...")
    calculates the flux within a small annulus centered at the star
    also calculates the median background
    '''
    a, b = image.shape
    n = box_size
    y,x = np.ogrid[-a/2.0:n-a/2.0, -b/2.0:n-b/2.0]
    mask = x*x + y*y <= r*r
    #square mask
    # x_min, x_max = 140, 160 #width of 20 pix
    # y_min, y_max= 140, 160
    # slice_mask = (x > x_min) * (x < x_max) * (y > y_min) * (y < y_max)
    # data_slice = data[y_min:y_max, x_min:x_max]
    center_mask = image[mask]
    with np.errstate(invalid='ignore'):
        peak_flux = np.median(center_mask) #star flux
        background = np.median(image)
    peak_flux_adjusted = peak_flux/header['NDR']/header['COADD']
    background_adjusted = background/header['NDR']/header['COADD']
    #print(peak_flux_adjusted)
    return peak_flux_adjusted, background_adjusted

def get_crop(image, centroid, box_size):
    '''
    simple cropping tool
    '''
    x, y = centroid
    image_crop = np.copy(image[int(y-(box_size/2)):int(y+(box_size/2)),int(x-(box_size/2)):int(x+(box_size/2))])
    return image_crop

def gauss(x, *params):
    A, mu, sigma, eps= params
    return A*np.exp(-(x-mu)**2/(2.*sigma**2)) + eps

def get_centroid(image):
    '''
    Calculate the centroid of a 2D array as its 'center of mass' determined from image moments.
    '''
    centroid = com(image)
    return centroid

def get_sources(image, fwhm, constant):
    daofind = DAOStarFinder(fwhm=fwhm, threshold=10*np.std(image))
    sources = daofind(image-constant)
    df = sources.to_pandas()
    return df

def check_psf(image,constant=1000,fwhm=20):
    centers=[]
    fig,ax=plt.subplots(1,1)

    df=get_sources(strip, fwhm, constant)
    new_centroid=df.sort_values(by='flux',ascending=False).head(1)[['xcentroid','ycentroid']].values.flatten()
    #print(new_centroid)

    swath = np.arange(int(new_centroid[0])-15,int(new_centroid[0])+15,1)
    psf_swath = []
    for j in swath:
        line=img_crop[j]/np.max(img_crop[j])
        psf_swath.append(line)

    mean_psf_swath = np.mean(psf_swath,axis=0)
    ax.plot(mean_psf_swath, 'o',label=idx)
    centers.append(new_centroid)

    #mean_psf= np.mean(mean_psf_swath,axis=0)
    ax.plot(mean_psf_swath, 'k-',label='mean')
    ax.set_title('PSF of {}'.format(pf.open(pol[0])[0].header['OBJECT']))
    plt.legend()
    plt.show()
    return mean_psf_swath, centers

def check_psf_old(pol,centroid_left,skip_every,box_size=150,constant=1000,fwhm=20):
    centers=[]
    fig,ax=plt.subplots(1,1)
    for idx,i in tqdm(enumerate(pol[::skip_every])):
        #print(pf.open(i)[0].header['OBJECT'])
        img=np.copy(pf.open(i)[0].data)
        strip = np.copy(img[:,int(centroid_left[0]-(box_size/2)):int(centroid_left[0]+(box_size/2))])
        df=get_sources(strip, fwhm, constant)
        new_centroid=df.sort_values(by='flux',ascending=False).head(1)[['xcentroid','ycentroid']].values.flatten()
        #print(new_centroid)
        try:
            img_crop = get_crop(strip, new_centroid, box_size)
            #vmin,vmax=np.median(img_crop), 10*np.median(img_crop)
            #plt.imshow(img_crop,vmin=vmin,vmax=vmax)
            #apertures = CircularAperture(new_centroid, r=fwhm/2)
            #apertures.plot(color='red', lw=1.5)
            #plt.show()
            swath = np.arange(int(new_centroid[0])-15,int(new_centroid[0])+15,1)
            psf_swath = []
            for j in swath:
                line=img_crop[j]/np.max(img_crop[j])
                psf_swath.append(line)
        except:
            pass
        mean_psf_swath = np.mean(psf_swath,axis=0)
        ax.plot(mean_psf_swath, 'o',label=idx)
        centers.append(new_centroid)

    #mean_psf= np.mean(mean_psf_swath,axis=0)
    ax.plot(mean_psf_swath, 'k-',label='mean')
    ax.set_title('PSF of {}'.format(pf.open(pol[0])[0].header['OBJECT']))
    plt.legend()
    plt.show()
    return mean_psf_swath, centers

def fit_psf(pol,mean_psf, centers):
    ydata = np.copy(mean_psf)#np.mean(mean_psf,axis=0))
    xdata = np.array(range(len(ydata)))
    #estimate mean and standard deviation
    mean = np.mean(centers,axis=0)[0]
    sigma = np.std(ydata)
    #fitting
    eps =0.1
    popt, pcov = curve_fit(gauss, xdata, ydata, p0 = [1, mean, sigma, eps])
    FWHM=2*np.sqrt(2*np.log(2))*popt[2]
    #plot psf
    plt.plot(xdata,gauss(xdata, *popt), label='Gaussian fit')
    plt.plot(xdata,ydata,'ok', label='data (mean)')
    plt.title('PSF of {}'.format(pf.open(pol[0])[0].header['OBJECT']))
    plt.xlabel('FWHM={0:.2f} pix, $\sigma$={1:.2f}'.format(FWHM,popt[2]))
    plt.legend()
    plt.show()
    print('A: {}\nmu: {}\nsigma= {}\neps: {}'.format(*popt)) #popt[0],popt[1], popt[2], popt[3]
    return popt, pcov

def make_dark():
    return
def make_calflat():
    return
def mask_badpix():
    return
def make_skyflat():
    return
def make_flatfield():
    return
def make_sky():
    return
#interpolating for bad pixel
def register():
    return
    #measure position offset
    #shifting and combining images

#analyzing standard star
def estimate_limit_mag():
    return
