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
import argparse
import getpass
#from ircs import utils

def image_sorter(input_dir, save_list=True):
    '''
    sort images inside input_dir based on header['OBJECT']
    input_dir can be changed if needed
    '''
    file_list = glob(os.path.join(input_dir,'IRCA*.fits'))
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
    params = 'FRAMEID, DATA-TYP, OBJECT, EXP1TIME, COADD, D_MODE, I_SCALE, \
                I_DTHNUM, I_DTHPOS'
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

    print('\nOBJECT:\n{}\n'.format(set(obj_type), obj_type))

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

######################################################

data_dir = '.'

parser = argparse.ArgumentParser(description=
    'Sorts the .fits files into proper directories based on their header',
    usage='use "%(prog)s --help" for more information',
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--save_list',
                help='outputs text file of list per category',
                type=bool, default=True)
args = parser.parse_args()
save_list = args.save_list

if not os.path.exists(data_dir):
    os.makedirs(input_dir)
    print('Created: {}'.format(data_dir))

#main
image_sorter(data_dir, save_list)

print('\n-----------------------')
print('         DONE')
print('-----------------------\n')
