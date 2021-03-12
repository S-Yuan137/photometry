import sy_class
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter

cutoff = sys.argv[1]
sizeinput = int(sys.argv[2])
freNum = int(sys.argv[3])
cutoff_str = str(cutoff).replace('-','_')
path = f"C:/Users/Shibo/Desktop/COMAP-sem2/maps/before_astro_cal/maps{cutoff_str}"
filenames_un = sy_class.get_filename_full(path, 'fits')
onlynames = sy_class.get_filename_full(path, 'fits',1)
filenames = sy_class.sortbyIband(onlynames, filenames_un)


def plot_map(mapobj, HDUname, sizeinput):
    fig=plt.figure()
    plt.style.use('science')
    mat, w = mapobj.getHDU(HDUname)
    med = np.nanmedian(mat)
    step= 2e-3
    ax = fig.add_subplot(111,projection=w)
    # lim=np.arange(-4*step+ med, 4*step + med, step)
    # mat = median_filter(mat, size = sizeinput)

    lim=np.arange(-10*step+ med, 10*step + med, step/2)
    mat = gaussian_filter(mat, sizeinput)
    h = ax.contour(mat,lim,origin='lower', cmap='jet')
    plt.xlabel('RA')
    plt.ylabel('DEC')
    temp=str(mapobj.getHDU(4)['iband']['value'][0])+'-'+str(mapobj.getHDU(4)['iband']['value'][1])+' '+mapobj.getHDU(4)['iband']['unit']
    plt.title(temp)
    plt.title(f'feed = 1, cutoff = {cutoff}K' +', '+ temp + f'\nGaussian filter size = {sizeinput} pixel')
    # plt.title(f'feed = 1, cutoff = {cutoff}K' +', '+ temp )
    cb=plt.colorbar(h)
    cb.set_label('T/K')
    plt.show()

map1 = sy_class.AstroMap(filenames[freNum])
plot_map(map1, 'primary', sizeinput)
