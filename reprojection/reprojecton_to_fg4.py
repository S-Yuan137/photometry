from reproject import reproject_interp
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS

hdu1 = fits.open('fg4_Feeds1-2-3-5-6-8-9-10-11-12-13-14-15-16-17-18-19_Band0.fits')[0]
hdu2 = fits.open('m31cm6i_3min_ss_modified.fits')[0]


array, footprint = reproject_interp(hdu2, hdu1.header)

ax1 = plt.subplot(1,2,1, projection=WCS(hdu1.header))
ax1.imshow(array, origin='lower')
ax1.coords['ra'].set_axislabel('Right Ascension')
ax1.coords['dec'].set_axislabel('Declination')
ax1.set_title('1')

ax2 = plt.subplot(1,2,2, projection=WCS(hdu1.header))
ax2.imshow(hdu1.data, origin='lower')
ax2.coords['ra'].set_axislabel('Right Ascension')
ax2.coords['dec'].set_axislabel('Declination')
ax2.coords['dec'].set_axislabel_position('r')
ax2.coords['dec'].set_ticklabel_position('r')
ax2.set_title('2')
plt.show()

fits.writeto('m31cm6i_3min_ss_on_fg4.fits', array, hdu1.header, overwrite=True)