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


mapCheck1 = sy_class.AstroMap(f'C:/Users/Shibo/Desktop/COMAP-sem2/week13/AddFeedsMaps/Ref10_Feeds1-2-9-10-19_Band4_PCbest.fits')
# mapCheck1.showaper(M31['centre'], M31['size'], M31['theta'], 1.2, refmap)
# plt.show()

mapCheck2 = sy_class.AstroMap(f'C:/Users/Shibo/Desktop/COMAP-sem2/week13/AddFeedsMaps/Ref10_Feeds1-9-10-11-19_Band0_PCbest.fits')

sy_class.T_Tplot(mapCheck2,[mapCheck1],M31['centre'], M31['size'], M31['theta'], [4,4])