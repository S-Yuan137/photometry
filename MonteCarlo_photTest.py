from astropy.io import fits                                  #package for read/write/save fits files
from astropy.wcs import WCS
import os
import numpy as np
import sy_class

dir_fits = 'MonteCarloSimulations'
name_fits = 'fg4_Feeds0_Band0_test.fits'
fout = dir_fits + '/' + name_fits
if not os.path.exists(dir_fits):
    os.makedirs(dir_fits)
elif os.path.exists(fout):
    os.remove(fout)

   

temp_dir = f"C:/Users/Shibo/Desktop/COMAP-sem2/week9/maps/fg4_Feeds11_Band0.fits"
mapobj = sy_class.AstroMap(temp_dir)
m, w = mapobj.getHDU('primary')
mat = np.zeros(shape=m.shape)
grey=fits.PrimaryHDU(mat, header= w.to_header())
greyHDU=fits.HDUList([grey])
greyHDU.writeto(fout)

