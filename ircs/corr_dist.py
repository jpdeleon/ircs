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
#from pyraf import iraf

try:
    from astropy.io import fits as pf
except:
    import pyfits as pf
from tqdm import tqdm

from ircs import utils

config = utils.check_config()
home_dir = config[0]
input_dir = config[1]
distcorr_dir = config[3]

#iraf.task(distcor = os.path.join(home_dir,'distcor.cl'))

config = utils.check_config()
db_file=config[7]
#dbs_file = 'ircsAO188_20mas_distmap_20131118.dbs'

def distmap(obj):
    for i in obj:
        '''
        bug: no choice but to save output
        '''
        fname_out=pf.open(obj[0])[0].header['FRAMEID']
        fname_out=os.path.join(flat_dir,i[:-5]+'g.fits')
        import pdb; pdb.set_trace()
        iraf.geotran(i,fname_out,database=db_file)
        #iraf.geotran(fnamein,fnameout,database='ch4_2014Jan.db',transfor='ch4_2014Jan.dat')


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
