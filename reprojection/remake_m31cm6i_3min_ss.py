from reproject import reproject_interp
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord

k = 1.38064852e-23
c = 299792458
mJy = 1e-29
GHz = 1e9
pre_factor = 2*np.log(2)*c**2*mJy/(np.pi*k*GHz**2)

def mJy_Beam2T(mat, freq, hpbeam_arcmin):
    # half power beamwidth is 3'
    # frequency in units of GHz
    # beam_width = (hpbeam_arcmin*u.arcmin).to(u.rad)
    beam_width = (hpbeam_arcmin/60)*(np.pi/180)
    return mat*pre_factor/(beam_width**2*freq**2)


########## read old data ############
# old_path = 'M31maps/MESSIER_031_I_6cm_blh1989.fits'
old_path = 'm31cm6i_3min_ss.fits'
old_hdu = fits.open(old_path)[0]
old_hdr = old_hdu.header
# print(repr(old_hdr))
#####################################
mat = old_hdu.data[0,:,:] * 1.1760699635e-05 + 14372.50977
data = mJy_Beam2T(mat, 4.85, 3) 
data = data - np.full(data.shape, 23) # set ref. zero is 23K

c=SkyCoord(ra=old_hdr['CRVAL1']*u.degree,dec=old_hdr['CRVAL2']*u.degree,frame='fk4')
c2000 = c.transform_to('fk5')
# c2000.ra.degree
# c2000.dec.degree

hdr = fits.Header()

hdr.set('SIMPLE',value = 'T', comment= 'file conforms to standard FITS format')
hdr.set('BITPIX',value = -64)
hdr.set('NAXIS',value = 2)
hdr.set('NAXIS1', value = old_hdr['NAXIS1'])
hdr.set('NAXIS2', value = old_hdr['NAXIS2'])
# hdr.set('DATE-OBS', value= old_hdr['DATE-OBS'])
hdr.set('BSCALE', value = 1.1760699635E-05)
hdr.set('BZERO', value = 14372.50977)
hdr.set('BUNIT', value = old_hdr['BUNIT'])
hdr.set('OBSRA', value = old_hdr['OBSRA'])
hdr.set('OBSDEC', value = old_hdr['OBSDEC'])

hdr.set('CTYPE1', value = old_hdr['CTYPE1'])
# hdr.set('CRVAL1', value = old_hdr['CRVAL1'])
hdr.set('CRVAL1', value = c2000.ra.degree)
hdr.set('CDELT1', value = old_hdr['CDELT1'])
hdr.set('CRPIX1', value = old_hdr['CRPIX1'])
hdr.set('CROTA1', value = old_hdr['CROTA1']*180/np.pi)

hdr.set('CTYPE2', value = old_hdr['CTYPE2'])

hdr.set('CRVAL2', value = c2000.dec.degree)

hdr.set('CDELT2', value = old_hdr['CDELT2'])
hdr.set('CRPIX2', value = old_hdr['CRPIX2'])
hdr.set('CROTA2', value = old_hdr['CROTA2']*180/np.pi)

# hdr.set('BSCALE', value = old_hdr['BSCALE'])
# hdr.set('BSCALE', value = old_hdr['BSCALE'])
# hdr.set('BSCALE', value = old_hdr['BSCALE'])
# hdr.set('BSCALE', value = old_hdr['BSCALE'])
# hdr.set('BSCALE', value = old_hdr['BSCALE'])
hdu = fits.PrimaryHDU(data, header= hdr)
hdu.writeto('m31cm6i_3min_ss_modified.fits', overwrite= True)
