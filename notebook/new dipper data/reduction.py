#!/usr/bin/env python

from pyraf import iraf
import glob

file_list=glob.glob('IRCA*.fits')
file_list.sort()

ch1='[1:256,1:1024]' 	#o
ch2='[257:512,1:1024]' 	#e

iraf.task(distcor = 'distcor.cl')
geomap='ircs+ao188_20mas_distmap_20131118.dbs'

for filename in file_list:
  #flatfielding
  iraf.imar(filename,'/','calflat.fits',filename[:-5]+'f.fits')
  #geometric distortion correction
  iraf.distcor(filename,filename[:-5]+'fg.fits',geomap)
  a1=filename[:-5]+ch1
  b1=filename[:-5]+'fg_ch1'
  a2=filename[:-5]+ch2
  b2=filename[:-5]+'fg_ch2'
  iraf.imcopy(a1,b1)
  iraf.imcopy(a2,b2)
  #remove intermediary files
  iraf.imdel(filename[:-5]+'f.fits')
  iraf.imdel(filename[:-5]+'fg.fits')
