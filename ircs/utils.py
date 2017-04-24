#!/usr/bin/env python
from glob import glob
import os
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

input_dir = '/mnt/B838B30438B2C124/data/ircs_pol'
output_dir = '/home/jpdl/ircs_pol_output'

## Data Reduction Pipeline
def check_header(fname):
    sample_hdr = pf.open(input_dir+'/'+fname)[0].header
    print(sample_hdr)
    return sample_hdr

def inspect_data():
    obj=[]
    pol = [] #Cyg, polarized
    unpol = [] #BD32, unpol
    off = []
    on = []

    file_list = glob(os.path.join(input_dir,'*.fits'))
    file_list.sort()
    if os.listdir(input_dir) != []:
        print('total no. of raw data frames: {0}\n'.format(len(file_list)))
    obj=[]
    for i in tqdm(file_list):
        hdr=pf.open(i)[0].header
        obj.append(hdr['OBJECT'])

        #classify objects
        if hdr['OBJECT'] == 'BD+32 3739':
            unpol.append(i)
        elif hdr['OBJECT'] == 'Cyg OB2 No.3':
            pol.append(i)
        elif hdr['OBJECT'].split()[0].split('_')[2] == 'OFF':
            #'IMAGE_Kp_OFF HWP0'
            #'IMAGE_Kp_OFF HWP22.5'
            #'IMAGE_Kp_OFF HWP45'
            #'IMAGE_Kp_OFF HWP67.5'
            off.append(i)

        else:
            #'IMAGE_Kp_ON HWP0',
            #'IMAGE_Kp_ON HWP22.5',
            #'IMAGE_Kp_ON HWP45',
            #'IMAGE_Kp_ON HWP67.5
            on.append(i)
    print('Objects based on header:\n')
    obj=set(obj)
    print(['{}'.format(i) for i in obj])
    print('\nNumber of frames per category:')
    #how to differentiate which is polarized and which is not?

    print('polarized?={0} \nunpolarized standard={1}\non={2}\noff={3}'.
      format(len(pol),len(unpol),len(on),len(off)))
    return file_list, obj, pol, unpol, on, off

def test_image(pol, unpol, on, off):
    #Cyg
    #Cyg
    pol_image = pf.open(pol[0])[0].data
    unpol_image = pf.open(unpol[0])[0].data

    on_image = pf.open(on[0])[0].data
    off_image = pf.open(off[0])[0].data

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

def get_crop(image, centroid, box_size):
    x, y = centroid
    image_crop = np.copy(image[int(y-(box_size/2)):int(y+(box_size/2)),int(x-(box_size/2)):int(x+(box_size/2))])
    return image_crop

def gauss(x, *params):
    A, mu, sigma, eps= params
    return A*np.exp(-(x-mu)**2/(2.*sigma**2)) + eps

def get_sources(image, fwhm, constant):
    daofind = DAOStarFinder(fwhm=fwhm, threshold=10*np.std(image))
    sources = daofind(image-constant)
    df = sources.to_pandas()
    return df

def check_psf(pol,centroid_left,skip_every,box_size=150,constant=1000,fwhm=20):
    centers=[]
    fig,ax=plt.subplots(1,1)
    for idx,i in tqdm(enumerate(pol[::skip_every])):
        #print(pf.open(i)[0].header['OBJECT'])
        img=pf.open(i)[0].data
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

def get_centroid(image):
    '''
    Calculate the centroid of a 2D array as its 'center of mass' determined from image moments.
    '''
    centroid = com(image)
    return centroid

get_median():
    return

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
