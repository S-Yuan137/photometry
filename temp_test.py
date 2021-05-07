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


###### do the photometry of 5C3.50 ########################
'''
path = f"C:/Users/Shibo/Desktop/COMAP-sem2/week10/maps"

filenames_un = sy_class.get_filename_full(path, 'fits')
onlynames = sy_class.get_filename_full(path, 'fits',1)
filenames = sy_class.sortbyIband(onlynames, filenames_un)
# mapnames = [sy_class.get_name_fromPath(fname) for fname in filenames]
mapobjs = []
for onefile in filenames:
    mapobjs.append(sy_class.AstroMap(onefile))

sy_class.plot_map(mapobjs, 'primary')
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
'''
##################################################################

############### sigma_f^2 percentage cut off #####################

# NoiseAper = {'centre':np.array([12.7452212, 40.4682106]), 'size': np.array([17, 5]), 'theta' : 0}
# feed = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19]
# for i in feed:
#     path = f'C:/Users/Shibo/Desktop/COMAP-sem2/week13/maps_sig_cuts_ref15/feed{i}_band0'
#     mapList = sy_class.getMapList(path, 'attrVal')
#     std_p, std_c = sy_class.NoiseStd(mapList, NoiseAper['centre'], NoiseAper['size'], NoiseAper['theta'])
#     # print([mapone.getPara('name') for mapone in mapList])
#     # print(std_p)
    
#     # print(','.join(map(str, std_c)))
#     pc = np.array([20,40,60,80,100])
#     print(stats_tools.findMinimaPoint(pc, std_p))

##################### Convergence test ############################

# M31 ={'centre':np.array([10.6836, 41.2790]), 'size':np.array([60,20]), 'theta':127}
# path = f"C:/Users/Shibo/Desktop/COMAP-sem2/week11/convTest/convTest"
# refmap = sy_class.AstroMap('C:/Users/Shibo/Desktop/COMAP-sem2/week11/m31cm6i_3min_ss_on_fg4.fits')
# filenames_un = sy_class.get_filename_full(path, 'fits')
# onlynames = sy_class.get_filename_full(path, 'fits',1)
# filenames = sy_class.sortbyIband(onlynames, filenames_un)

# mapobjs = []
# for onefile in filenames:
#     mapobjs.append(sy_class.AstroMap(onefile))

# rms = []
# for onemap in mapobjs:
#     _,conv,_ = sy_class.jackknife(onemap,[240,240],40)
#     print(onemap.getPara('name'))
#     print(sy_class.jackknife(onemap,[240,240],40))
#     rms.append(round(conv,4))
# sy_class.plot_map(mapobjs, 'primary')
# sy_class.T_Tplot(refmap, mapobjs, M31['centre'], M31['size'], M31['theta'], [True, 4, 4])
# plt.show()



########################### temp use for check add feeds maps ##############################
refmap = sy_class.AstroMap('C:/Users/Shibo/Desktop/COMAP-sem2/week11/m31cm6i_3min_ss_on_fg4.fits')
refmapfull = sy_class.AstroMap('C:/Users/Shibo/Desktop/COMAP-sem2/week11/m31cm6i_3min_full_on_fg4.fits')

temp_refmap = sy_class.AstroMap('C:/Users/Shibo/Desktop/COMAP-sem2/week13/AddFeedsMaps/Ref10_FeedsAll-8_Band0_PCbest.fits')
temp_refmap2 = sy_class.AstroMap('C:/Users/Shibo/Desktop/COMAP-sem2/week13/AddFeedsMaps/Ref10_FeedsAll_Band0_PCbest.fits')

M31 ={'centre':np.array([10.6836, 41.2790]), 'size':np.array([60,20]), 'theta':127}
RG5C3_50 = {'centre':np.array([9.6076856,41.6096426]), 'size':np.array([6,6]), 'theta':0}
NoiseAper = {'centre':np.array([12.7452212, 40.4682106]), 'size': np.array([17, 5]), 'theta' : 0}

maplist = []
for feed in [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19]:
    path = f'C:/Users/Shibo/Desktop/COMAP-sem2/week13/maps_sig_cuts_BestPC/feed{feed}_band0'
    maplist.extend(sy_class.getMapList(path, 'attrVal'))

print(sy_class.NoiseStd([temp_refmap,temp_refmap2], NoiseAper['centre'], NoiseAper['size'], NoiseAper['theta']))
# maplist[0].showaper(M31['centre'], M31['size'],M31['theta'], 1.2, refmap)
# temp_refmap.showaper(M31['centre'], M31['size'],M31['theta'], 1.2, refmap)

# for onemap in maplist:
#     onemap.showaper(M31['centre'], M31['size'],M31['theta'], 1.2, refmap)
    # sy_class.T_Tplot(temp_refmap, [onemap], M31['centre'], M31['size'],M31['theta'], [5,5],1.2)
    # plt.show()

temp_refmap.showaper(M31['centre'], M31['size'],M31['theta'], 1.2, refmapfull)
# sy_class.plot_map(maplist, 'primary')
# sy_class.T_Tplot(temp_refmap, [temp_refmap2], M31['centre'], M31['size'],M31['theta'], [5,5],1.2)
# sy_class.T_Tplot(temp_refmap, [maplist[0]], RG5C3_50['centre'], RG5C3_50['size'], RG5C3_50['theta'], [2,2],1.2)

plt.show()
