#!/usr/bin/env python

from glob import glob 
import sys
import pyraf
import numpy as np
from pyraf import *
from csv import *

open1 = str(sys.argv[1])
olist = open(open1,"r")
framename = np.loadtxt(olist,dtype=str)

iraf.task(distcor = '/home/Jerome/data/imaging/ircs_UH30B/TU/distcor.cl')
f1 = open('gch1_'+open1,"w") ### output frame
f2 = open('gch2_'+open1,"w") ### output frame

flat = str(sys.argv[2]) ### flat frame

geodata = str(sys.argv[3]) ### input list - distortion map


for i in range(len(framename)):
 iraf.imar(framename[i],'/',flat,'fl_'+framename[i]) ### flat fielding
 newdata = 'fl_'+framename[i]
 iraf.distcor(newdata,'g_'+framename[i],geodata) ### distortion correction
 iraf.imcopy('g_ch1'+framename[i]+'[1:256,1:1024]','g_'+framename[i]) ### cutting needless regions
 iraf.imcopy('g_ch2'+framename[i]+'[256:512,1:1024]','g_'+framename[i]) 

 iraf.imdel('fl_'+framename[i])
 iraf.imdel('g_'+framename[i])
 print >> f1, 'g_ch1'+framename[i] ### write list
 print >> f2, 'g_ch2'+framename[i] ### write list
