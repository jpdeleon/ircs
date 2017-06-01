### Warning ###
# numpy 1.12.* does not work
###############

# T.U made for centering dithering images 
# May 3 2017

#### usage ####
# python getcentroid.py image.lst
###############

import numpy as np
import sys
import pyfits
import pyraf
from pyraf import iraf
import subprocess
import os
import scipy.ndimage.interpolation as ndinter

open1 = str(sys.argv[1]) ### your frame list
framelist = np.loadtxt(open1,dtype=str)

# iraf setting #
boxarea = 5
iraf.fitpsf.function='elgauss'; iraf.fitpsf.maxiter=1000; iraf.fitpsf.interactive='no'; iraf.fitpsf.verify='no'; iraf.fitpsf.update='no'; iraf.fitpsf.verbose='no'

for i in range(len(framelist)):

 print "processing the data "+framelist[i]
 out1 = open('ini.dat',"w")
 #out2 = open('sexresult_'+framelist[i].split(".")[0].split("_")[2]+'.dat',"w")

 cmd = "sex %s" % (framelist[i]) # the same as 'sex XXXXX.fits'
 subprocess.call( cmd.strip().split(" ") )
 os.system('sort -rg result.cat > result_sort.cat')

 sex_result = np.loadtxt('result_sort.cat',dtype=str)
 if np.ndim(sex_result) > 1:
  xcen_sex = sex_result[0][4] # Sextractor's centroid of the PSF
  ycen_sex = sex_result[0][5]
 
 else: # sextractor detects only the target's PSF
  xcen_sex = sex_result[4]
  ycen_sex = sex_result[5]

 print >> out1, xcen_sex, ycen_sex
 out1.close()
 #print >> out2, sex_result
 #out2.close()

 iraf.fitpsf(framelist[i],box=boxarea,coords='ini.dat')

 xcen=float(iraf.pdump(framelist[i]+'.psf.1',fields='xcenter',Stdout=1,expr='yes')[0]) # IRAF's centroid of the PSF
 ycen=float(iraf.pdump(framelist[i]+'.psf.1',fields='ycenter',Stdout=1,expr='yes')[0])

 # debug
 #delta_x = float(xcen_sex) - xcen
 #delta_y = float(ycen_sex) - ycen
 #print >> out2, framelist[i], delta_x, delta_y

 shift_x = 513.0 - xcen #IRAF shift
 shift_y = 513.0 - ycen

 im = pyfits.open(framelist[i])[0].data
 hdr = pyfits.getheader(framelist[i])
 hdr['CRPIX1'] = 513 # center of the extended frames in IRAF or ds9
 hdr['CRPIX2'] = 513

 dummy=np.zeros((1024,1024),"f") # extending the frame
 dummy[:,:]=np.NAN
 
 dummy[shift_y:512+shift_y,shift_x:512+shift_x] = im # shifting

 pyfits.writeto(framelist[i].split(".")[0]+'_r.'+framelist[i].split(".")[1],dummy,hdr)

#out2.close()
