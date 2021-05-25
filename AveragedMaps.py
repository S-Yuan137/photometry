import sy_class
import stats_tools
import numpy as np
import matplotlib.pyplot as plt
################ average maps ####################################################################
'''
# feed = [1,2,3,5,6,9,10,11,12,13,14,15,16,17,18,19]
feed = [1, 2, 9, 10, 19]
Band = 4


PC = 'best'
Ref = 10

maplist = []
for i in feed:
    path = f'C:/Users/Shibo/Desktop/COMAP-sem2/week14/maps_band4_BestPC/feed{i}_band{Band}'
    # path = f'C:/Users/Shibo/Desktop/COMAP-sem2/week13/maps_sig_cuts_ref10/feed{i}_band0'
    maplist.extend(sy_class.getMapList(path, 'attrVal'))
    # maplist.append(AstroMap(f'C:/Users/Shibo/Desktop/COMAP-sem2/week13/maps_sig_cuts_ref{Ref}/feed{i}_band0/fg4_Feeds{i}_Band0_PC{PC}.fits'))
feedstr = '-'.join(map(str, feed))
sy_class.WeightAverageMap(maplist, f'Ref{Ref}_Feeds{feedstr}_Band{Band}_PC{PC}.fits', 'C:/Users/Shibo/Desktop/COMAP-sem2/week13/AddFeedsMaps')
# WeightAverageMapVersion2(maplist, f'Ref{Ref}_FeedsAll_Band0_PCAll.fits', 'C:/Users/Shibo/Desktop/COMAP-sem2/week13/AddFeedsMaps/AddFeedsMapsVersion2')
'''


################### check maps ######################################################################

refmap = sy_class.AstroMap('C:/Users/Shibo/Desktop/COMAP-sem2/week11/m31cm6i_3min_ss_on_fg4.fits')
refmaplarge = sy_class.AstroMap('C:/Users/Shibo/Desktop/COMAP-sem2/week11/m31cm6i_3min_full_on_fg4.fits')

NoiseAper = {'centre':np.array([12.7452212, 40.4682106]), 'size': np.array([17, 5]), 'theta' : 0}
M31 ={'centre':np.array([10.6836, 41.2790]), 'size':np.array([60,20]), 'theta':127}
M31part = {'centre':np.array([11.0512218, 41.3032980]), 'size':np.array([30,15]), 'theta':120}
M31part2 = {'centre':np.array([10.2352350, 41.0097024]), 'size':np.array([24,8]), 'theta':110}
RG5C3_50 = {'centre':np.array([9.6076856,41.6096426]), 'size':np.array([6,6]), 'theta':0}

mapCheck1 = sy_class.AstroMap(f'C:/Users/Shibo/Desktop/COMAP-sem2/week13/AddFeedsMaps/Ref10_Feeds1-2-9-10-19_Band4_PCbest.fits')
# mapCheck1 = sy_class.AstroMap(f'C:/Users/Shibo/Desktop/COMAP-sem2/week14/maps_sig_cuts_allfeed_PCcut/fg4_Feeds1-2-3-5-6-8-9-10-11-12-13-14-15-16-17-18-19_Band0_PC60.fits')
# print(sy_class.NoiseStd([mapCheck1], NoiseAper['centre'], NoiseAper['size'], NoiseAper['theta']))
# mapCheck1.showaper(M31['centre'], M31['size'], M31['theta'], 1.2, refmap)
# plt.show()

mapCheck2 = sy_class.AstroMap(f'C:/Users/Shibo/Desktop/COMAP-sem2/week13/AddFeedsMaps/Ref10_Feeds1-9-10-11-19_Band0_PCbest.fits')
# mapCheck1.showaper(M31['centre'], M31['size'], M31['theta'], 1.2, refmap)
# plt.show()
# sy_class.T_Tplot(refmap,[mapCheck1],M31['centre'], M31['size'], M31['theta'], [4,4])
# print(sy_class.NoiseStd([mapCheck2], NoiseAper['centre'], NoiseAper['size'], NoiseAper['theta']))

print(sy_class.photometry(mapCheck2, M31['centre'], 60, 20, M31['theta'], 2))
print(sy_class.photometry(mapCheck1, M31['centre'], 60, 20, M31['theta'], 2))

# print(sy_class.photometry(mapCheck2, RG5C3_50['centre'], 6, 6, RG5C3_50['theta'], 2))
#26.5 GHz 6.96 0.06
#30.5     8.87 0.07

# print(stats_tools.s_index(obj))
fre=np.array([26.5, 30.5])
flux=np.array([6.96, 8.87])
flux_err=np.array([0.06, 0.07])
lnfre=np.log(fre)
lnflux=np.log(flux)
lnflux_err=np.abs(flux_err/flux)
alpha_value=stats_tools.ordinary_least_squares(lnfre,lnflux)[0]
alpha_err=stats_tools.ordinary_least_squares_err(lnfre,lnflux,lnflux_err)[0]
alpha=np.array([alpha_value,alpha_err])
print(alpha)