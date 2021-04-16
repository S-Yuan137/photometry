import sy_class
import stats_tools
import matplotlib.pyplot as plt
import sys
import numpy as np

def Source(name, mapobjList, centre, size, theta, width):
    freq = []
    flux = []
    flux_err = []
    for onemap in mapobjList:
        result = sy_class.photometry(onemap, centre, size[0], size[1], theta, width)
        freq.append(np.mean(onemap.getPara('freq')))
        flux.append(result['aper_sum_bkgsub_Jy'])
        flux_err.append(result['bkg_rms_Jy'])
    return {'name':name, 'freq':np.array(freq),'flux':np.array(flux),'flux_err':np.array(flux_err)[:,0]}



path = f"C:/Users/Shibo/Desktop/COMAP-sem2/week10/maps"

filenames_un = sy_class.get_filename_full(path, 'fits')
onlynames = sy_class.get_filename_full(path, 'fits',1)
filenames = sy_class.sortbyFeed(onlynames, filenames_un)
mapnames = [sy_class.get_name_fromPath(fname) for fname in filenames]
mapobjs = []
for onefile in filenames:
    mapobjs.append(sy_class.AstroMap(onefile))
print(sy_class.jackknife(mapobjs[0],[240,240],40))
centre = np.array([9.6076856,41.6096426])
size = np.array([6,6])
theta = 0
width = 2

obj = Source('5C3.50',mapobjs,centre, size, theta, width)
print(stats_tools.s_index(obj))
print(stats_tools.chi_sqare(obj))
sy_class.sp_plot(obj)
sy_class.fitting_plot(obj)
plt.title('5C3.50')
# plt.legend()
plt.show()
