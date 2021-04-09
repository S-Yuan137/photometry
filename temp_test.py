import sy_class
import matplotlib.pyplot as plt
import sys
import numpy as np

path1 = f"C:/Users/Shibo/Desktop/COMAP-sem2/week7/fg4_Feeds1-11-14-15-17-18_Band0_half0.fits"
path2 = f"C:/Users/Shibo/Desktop/COMAP-sem2/week7/fg4_Feeds1-11-14-15-17-18_Band0_half1.fits"


map1 = sy_class.AstroMap(path1)
map2 = sy_class.AstroMap(path2)
aperlist = np.arange(5,40,1)
std =[]
centre =  [240,240]
for aper in aperlist:
    std.append(sy_class.jackknife(map1, centre,aper,map2)[1])

fig = plt.figure()
h=plt.plot(aperlist,std, '*')
# plt.xticks(np.arange(0,17),labels=feeds)
plt.xlabel('Aperture radius/arcmin')
# plt.yticks(np.arange(0,17),labels=feeds)
plt.ylabel('STD/K')
plt.title('Jackknife: STD')


sy_class.plot_diffmap(map1,map2,centre, aperlist[-1])

plt.show()

