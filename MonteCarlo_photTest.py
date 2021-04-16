from astropy.io import fits                                  #package for read/write/save fits files
from astropy.wcs import WCS
import os
import numpy as np
import sy_class
from photutils.datasets import make_100gaussians_image

########### generate the map #################
'''
 id xcenter ycenter aperture_sum annulus_median  aper_bkg aper_sum_bkgsub
      pix     pix
--- ------- ------- ------------ -------------- --------- ---------------
  1   145.1   168.3    1131.5794       4.848213 380.77776       750.80166
  2    84.5   224.1    746.16064      5.0884354 399.64478       346.51586
  3    48.3   200.3    1250.2186      4.8060599 377.46706        872.7515
'''

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

mat = make_100gaussians_image()
grey=fits.PrimaryHDU(mat, header= w.to_header())
greyHDU=fits.HDUList([grey])
greyHDU.writeto(fout)



############ photometry test #################
pos_deg = [12.72811659,40.0719062]
        #   [14.09437465,40.96973421]
        #   [14.86719693,40.54710286]
path = 'MonteCarloSimulations/fg4_Feeds0_Band0_test.fits'
mapobj = sy_class.AstroMap(path)
print(sy_class.photometry(mapobj,pos_deg,5,5,0,5)[-1])
