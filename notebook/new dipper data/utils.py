import sys
import numpy as np
try:
    from astropy.io import fits as pf
except:
    import pyfits as pf
import matplotlib.pyplot as plt
from tqdm import tqdm


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
