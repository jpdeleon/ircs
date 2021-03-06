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

#unpack
config = utils.check_config()
home_dir = config[0]
input_dir = config[1]
output_dir = config[2]
crop_output_dir = config[3]

if not os.path.exists(crop_output_dir):
    os.makedirs(crop_output_dir)

#width of o-,e-strip is 4.4 arcsec
strip_width = 4.4
ircs_pix_size = 20.57*1e-3
separation = strip_width/ircs_pix_size


#these should be edited from config file
centroid_dither_1_left = (620,520)
centroid_dither_1_right = (880,520)
centroid_dither_2_left = (620,230)
centroid_dither_2_right = (880,230)
centroid_dither_3_left = (620,380)
centroid_dither_3_right = (880,380)
centroid_dither_4_left = (620,670)
centroid_dither_4_right = (880,670)
centroid_dither_5_left = (620,820)
centroid_dither_5_right = (880,820)

xleft = (512,768)
yleft = (1,1024)
xright = (768,1024)
yright = (1,1024)

image_o = {} #o-images
header_o = {}
image_e = {} #o-images
header_e = {}

saturated_o = []
saturated_e = []

def extract_oe(obj, show_oe_image, save_fits, check_if_saturated, cmap):
    dither_step=pf.open(obj[0])[0].header['I_DTHSZ']
    dither_step_in_pix = int(dither_step/ircs_pix_size)
    for i in obj:
        dither = pf.open(i)[0].header['I_DTHPOS']
        dither_position = dither.split(':')[0].strip()
        hdr_l = pf.open(i)[0].header
        hdr_r = pf.open(i)[0].header #np.copy(hdr_l)
        '''
        bug: add comment in fits header
        comment = 'estimated centroid'
        '''
        image = np.copy(pf.open(i)[0].data)
        #cropp: 4."4 x 21"
        image_crop_l = image[yleft[0]:yleft[1],xleft[0]:xleft[1]]
        image_crop_r = image[yright[0]:yright[1],xright[0]:xright[1]]

        image_o[dither] = image_crop_l
        image_e[dither] = image_crop_r
        #header_e and header_e have different estimated centroid at least
        header_o[dither] = hdr_l
        header_e[dither] = hdr_r

        if dither_position == '1':
            hdr_l['centroid'] = str(centroid_dither_1_left) #comment
            hdr_r['centroid'] = str(centroid_dither_1_right)
        elif dither_position == '2':
            hdr_l['centroid'] = str(centroid_dither_2_left)
            hdr_r['centroid'] = str(centroid_dither_2_right)
        elif dither_position == '3':
            hdr_l['centroid'] = str(centroid_dither_3_left)
            hdr_r['centroid'] = str(centroid_dither_3_right)
        elif dither_position == '4':
            hdr_l['centroid'] = str(centroid_dither_4_left)
            hdr_r['centroid'] = str(centroid_dither_4_right)
        else: #dither_position == '5':
            hdr_l['centroid'] = str(centroid_dither_5_left)
            hdr_r['centroid'] = str(centroid_dither_5_right)

        if show_oe_image == True:
            '''
            loops all images and display them; 'status' is for exiting the display mode
            however, q is entered, images will not be saved, so it is advised to edited
            show_image=False just to skip showing image
            '''
            import pdb; pdb.set_trace()
            status=utils.compare_oe(image_o[dither], image_e[dither], header_o[dither], header_e[dither], cmap)


            if status == False:
                return # acts like 'break' to exit displaying all images
        if save_fits == True:
            try:
                pf.writeto(os.path.join(crop_output_dir,header_o[dither]['FRAMEID'],'_o.fits'), image_o[dither], header_o[dither])
            except:
                print('{}_o.fits already exists!'.format(header_o[dither]['FRAMEID']))
            try:
                pf.writeto(os.path.join(crop_output_dir,header_e[dither]['FRAMEID'],'_e.fits'), image_e[dither], header_e[dither])
            except:
                print('{}_e.fits already exists!'.format(header_o[dither]['FRAMEID']))
        if check_if_saturated == True:
            '''
            create a mask in the center (within star) and get its median
            '''
            #circular mask centered at centroid/ star
            #mask for image_e is the same
            try:
                peak_flux_o, bg_o = utils.get_peak_flux(image_o[dither], header_o[dither], box_size, r=3)
                if  peak_flux_o > 4000:
                    print('\nFLAG:\n{} is saturated (peak count > 4000 ADU)!\n'.format(i))
                    saturated_o.append(i)
            except:
                print('ERROR encountered in {}'.format(header_o[dither]['FRAMEID']))
            try:
                peak_flux_e, bg_e = utils.get_peak_flux(image_e[dither], header_e[dither], box_size, r=3)
                if  peak_flux_e > 4000:
                    print('\nFLAG:\n{} is saturated (peak count > 4000 ADU)!\n'.format(i))
                    saturated_e.append(i)
            except:
                print('ERROR encountered in {}'.format(header_e[dither]['FRAMEID']))

    print('\nTotal number of saturated frames:\n{}'.format(len(saturated_o)))
    #return None
