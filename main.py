import sy_class
import matplotlib.pyplot as plt
import sys
import numpy as np


# path = f"C:/Users/Shibo/Desktop/COMAP-sem2/maps/before_astro_cal/maps{cutoff_str}"
path = f"C:/Users/Shibo/Desktop/COMAP-sem2/maps/after_astro_cal/maps/band1"
filenames_un = sy_class.get_filename_full(path, 'fits')
onlynames = sy_class.get_filename_full(path, 'fits',1)
# filenames = sy_class.sortbyIband(onlynames, filenames_un)
filenames = sy_class.sortbyFeed(onlynames, filenames_un)

mapnames = [sy_class.get_name_fromPath(fname) for fname in filenames]
mapobjs = []
for onefile in filenames:
    mapobjs.append(sy_class.AstroMap(onefile))

mean_mat = np.zeros(shape=(len(filenames),len(filenames)))
std_mat = np.zeros(shape=(len(filenames),len(filenames)))
feeds = []
for i, onemap_i in enumerate(mapobjs):
    feeds.append(onemap_i.getPara('feed'))
    for j, onemap_j in enumerate(mapobjs):
        mean_mat[i,j] = sy_class.jackknife(onemap_i,[240,240],10,onemap_j)[0]
        std_mat[i,j] = sy_class.jackknife(onemap_i,[240,240],10,onemap_j)[1]
feeds = np.array(feeds)
fig = plt.figure()
h=plt.imshow(abs(mean_mat), cmap='jet')
plt.xticks(np.arange(0,17),labels=feeds)
plt.xlabel('Feeds')
plt.yticks(np.arange(0,17),labels=feeds)
plt.ylabel('Feeds')
plt.title('Jackknife: Mean')
cb=plt.colorbar(h)
cb.set_label('T/K')
plt.show()

fig = plt.figure()
h=plt.imshow(abs(std_mat), cmap='jet')
plt.xticks(np.arange(0,17),labels=feeds)
plt.xlabel('Feeds')
plt.yticks(np.arange(0,17),labels=feeds)
plt.ylabel('Feeds')
plt.title('Jackknife: STD')

cb=plt.colorbar(h)
cb.set_label('T/K')
plt.show()
