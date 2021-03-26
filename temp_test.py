import sy_class
import matplotlib.pyplot as plt
import sys
import numpy as np

# path1 = f"C:/Users/Shibo/Desktop/COMAP-sem2/maps/after_astro_cal/fg4_Feeds1-11-14-15-17-18_Band0.fits"
# path2 = f"C:/Users/Shibo/Desktop/COMAP-sem2/maps/after_astro_cal/fg4_Feeds1-2-3-5-6-8-9-10-11-12-13-14-15-16-17-18-19_Band0.fits"
path1 = f"C:/Users/Shibo/Desktop/COMAP-sem2/maps/after_astro_cal/maps/band0/fg4_Feeds16_Band0.fits"
path2 = f"C:/Users/Shibo/Desktop/COMAP-sem2/maps/after_astro_cal/maps/band0/fg4_Feeds1_Band0.fits"

map1 = sy_class.AstroMap(path1)
map2 = sy_class.AstroMap(path2)
aperlist = np.arange(5,80,1)
std =[]
for aper in aperlist:
    std.append(sy_class.jackknife(map1, [240,240],aper, map2)[1])

fig = plt.figure()
h=plt.plot(aperlist,std, '*')
# plt.xticks(np.arange(0,17),labels=feeds)
plt.xlabel('Aperture radius/arcmin')
# plt.yticks(np.arange(0,17),labels=feeds)
plt.ylabel('STD/K')
plt.title('Jackknife: STD')


sy_class.plot_diffmap(map1,map2,'primary')

plt.show()

