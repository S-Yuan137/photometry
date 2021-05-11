import sy_class
import stats_tools
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys
import numpy as np

######################## the feed-feed jackknife plot #####################################
'''
path = f"C:/Users/Shibo/Desktop/COMAP-sem2/week10/EachFeedmaps/maps" # this is the all dataset
mapobjs = sy_class.getMapList(path, 'feed')


# aper_size = 40
mean_mat = np.zeros(shape=(len(mapobjs),len(mapobjs)))
std_mat = np.zeros(shape=(len(mapobjs),len(mapobjs)))
feeds = []
for i, onemap_i in enumerate(mapobjs):
    feeds.append(onemap_i.getPara('feed'))
    for j, onemap_j in enumerate(mapobjs):
        if i != j:
            mean_mat[i,j], std_mat[i,j] = sy_class.jackknife(onemap_i,[180,280],40,onemap_j)
            # std_mat[i,j] = (sy_class.jackknife(onemap_i,[240,240],40,onemap_j)[1]+
            #                             sy_class.jackknife(onemap_i,[180,280],40,onemap_j)[1]+
            #                             sy_class.jackknife(onemap_i,[180,200],40,onemap_j)[1]+
            #                             sy_class.jackknife(onemap_i,[300,280],40,onemap_j)[1]+
            #                             sy_class.jackknife(onemap_i,[300,200],40,onemap_j)[1])/5
            # mean_mat[i,j] = (sy_class.jackknife(onemap_i,[240,240],40,onemap_j)[0]+
            #                             sy_class.jackknife(onemap_i,[180,280],40,onemap_j)[0]+
            #                             sy_class.jackknife(onemap_i,[180,200],40,onemap_j)[0]+
            #                             sy_class.jackknife(onemap_i,[300,280],40,onemap_j)[0]+
            #                             sy_class.jackknife(onemap_i,[300,200],40,onemap_j)[0])/5
        else:
            mean_mat[i,j] = np.nan
            std_mat[i,j] = np.nan
        
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
'''

############### feed-feed Pearson correlation coefficient ###################

# M31 ={'centre':np.array([10.6836, 41.2790]), 'size':np.array([60,20]), 'theta':127}
# path = f"C:/Users/Shibo/Desktop/COMAP-sem2/week10/EachFeedmaps/maps" # this is the all dataset
# mapobjs = sy_class.getMapList(path, 'feed')

feeds = np.array([1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19])
mapobjs = []
for i in feeds:
    path = f'C:/Users/Shibo/Desktop/COMAP-sem2/week13/maps_sig_cuts_BestPC/feed{i}_band0'
    mapobjs.extend(sy_class.getMapList(path, 'attrVal'))
    # mapobjs.append(sy_class.AstroMap(f'C:/Users/Shibo/Desktop/COMAP-sem2/week13/maps_sig_cuts_ref10/feed{i}_band0/fg4_Feeds{i}_Band0_PC100.fits'))
    # maplist.append(AstroMap(f'C:/Users/Shibo/Desktop/COMAP-sem2/week13/maps_sig_cuts_ref{Ref}/feed{i}_band0/fg4_Feeds{i}_Band0_PC{PC}.fits'))


centre_world = np.array([10.6836, 41.2790])
size = np.array([60,20])
theta_deg= 127
downsample = [None, None]

PCC_r = np.zeros(shape=(len(mapobjs),len(mapobjs)))
PCC_p = np.zeros(shape=(len(mapobjs),len(mapobjs)))

for i, onemap_i in enumerate(mapobjs):
    # feeds.append(onemap_i.getPara('feed'))
    for j, onemap_j in enumerate(mapobjs):
        if i != j:
            pri_mat_i, wcs_header_i = onemap_i.getHDU('primary')
            pri_mat_j, wcs_header_j = onemap_j.getHDU('primary')
            x0_i, y0_i = wcs_header_i.wcs_world2pix(centre_world[0], centre_world[1], 0)
            x0_j, y0_j = wcs_header_j.wcs_world2pix(centre_world[0], centre_world[1], 0)
            pri_mat_i = stats_tools.maskCut(pri_mat_i, [x0_i, y0_i], size, theta_deg, downsample)
            pri_mat_i = pri_mat_i.flatten()
            pri_mat_j = stats_tools.maskCut(pri_mat_j, [x0_j, y0_j], size, theta_deg, downsample) 
            pri_mat_j = pri_mat_j.flatten()

            PCC_r[i,j], PCC_p[i,j] = pearsonr(pri_mat_i, pri_mat_j)
            
        else:
            PCC_r[i,j] = np.nan
            PCC_p[i,j] = np.nan
        

fig = plt.figure()
h=plt.imshow(PCC_r, cmap='jet')
plt.xticks(np.arange(0,17),labels=feeds)
plt.xlabel('Feeds')
plt.yticks(np.arange(0,17),labels=feeds)
plt.ylabel('Feeds')
plt.title('Pearson correlation coefficient')
cb=plt.colorbar(h)
# cb.set_label('T/K')
plt.show()

fig = plt.figure()
h=plt.imshow(PCC_p, cmap='jet')
plt.xticks(np.arange(0,17),labels=feeds)
plt.xlabel('Feeds')
plt.yticks(np.arange(0,17),labels=feeds)
plt.ylabel('Feeds')
plt.title('Pearson correlation coefficient: p-values')

cb=plt.colorbar(h)
# cb.set_label('T/K')
plt.show()