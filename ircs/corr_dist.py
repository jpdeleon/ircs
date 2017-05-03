#!/usr/bin/env python

#https://www.naoj.org/Observing/Instruments/IRCS/camera/IRCS+AO188_distortion/

'''
DrizzlePac from STScI (https://github.com/spacetelescope/drizzlepac)
contains methods that replicate (somewhat) geomap except they can do the
fitting only using linear transformations (no support for higher order
polynomials):

import drizzlepac
drizzlepac.linearfit.fit_all(xy, uv, method='rscale')

NOTE: method='general' allows fitting for skews as well (that is, it
produces two rotation angles, two scales, and two shifts).

Another low-level package https://github.com/spacetelescope/drizzle can be
used to resample your images. However, you will need to provide your own
coordinate mappings computed from the transformations found in the fitting
step.
'''

from glob import glob
import os
import numpy as np

try:
    from astropy.io import fits as pf
except:
    import pyfits as pf
from tqdm import tqdm

from ircs import utils

#input_dir = '/home/jp/data/ircs_pol'
input_dir = '/mnt/sda1/data/ircs_pol'
#input_dir = '/mnt/B838B30438B2C124/data/ircs_pol'

dbs_file = 'ircs+ao188_20mas_distmap_20131118.dbs'

def distmap(obj):

    for i in obj:
        fname_out=pf.open(obj[0])[0].header['FRAMEID']
        fname_out=filename1[:-5]+'g.fits'
        iraf.geotran(i,fname_out,database=dbs_file)#,transfor='ch1_2014Jan.dat')

# procedure distcor(inlist, outlist, database)
# 	  file inlist   {prompt='List of flat corrected images with @mark'}
# 	  file outlist  {prompt='List of output distortion corrected images with @mark'}
# 	  file database {prompt='Distortion solution in the IRAF geomap database format'}
# begin
# 	char _inlist, _outlist, _database
# 	char p1,p2
# 	char gmp
#
# 	_inlist = inlist
# 	_outlist = outlist
# 	_database = database
#
# 	list = _database
# 	gmp = 'none'
# 	while(fscan(list, p1, p2) != EOF){
# 		if (p1 == 'begin'){
# 		   gmp = p2
# 		}
# 	}
#
# 	if (gmp == 'none'){
# 	   print('no coodinate transforms found in the database')
# 	   exit
# 	}
#
# 	unlearn('geotran')
# 	geotran(inlist, outlist, database, gmp)
#
#
# end
