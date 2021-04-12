import stats_tools
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import os 
from astropy.io import fits                                  #package for read/write/save fits files
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from photutils import CircularAperture                      # package for define the aperture
from photutils import aperture_photometry                   # package for photometry
from photutils import EllipticalAperture
from photutils import EllipticalAnnulus
from astropy.stats import sigma_clipped_stats
from math import e

def iband2fre(Num):  # bandwidth, in units of GHz
    numbers = {
        0 : [26,27],
        1 : [27,28],
        2 : [29,30],
        3 : [28,29],
        4 : [30,31],
        5 : [31,32],
        6 : [33,34],
        7 : [32,33]
    }
    return numbers.get(int(Num), np.nan)

# avoid using np.argsort since the input arguments are lists, even if different length
def sort_list(list1,list2): #according to list1 to sort two lists
    zipped=zip(list1,list2)
    sort_zipped = sorted(zipped,key=lambda x:(x[0]))
    result = zip(*sort_zipped) # 将 sort_zipped 拆分成两个元组
    # list1_s, list2_s= [list(x) for x in result]
    list2_s= [list(x) for x in result][1]
    return list2_s

def sortbyIband(names, filenames): # defalut format is the last character of names is the iband number
    iband = [iname[-1] for iname in names]
    fre_start = [iband2fre(i)[0] for i in iband]
    return sort_list(fre_start, filenames)

def sortbyFeed(names, filenames):
    spliter = '_'
    feeds = []
    for onename in names:
        subStrs = onename.split(spliter,2)
        feeds.append([int(i) for i in subStrs[1][5:].split('-')])
    return sort_list(feeds, filenames)
    

def get_filename_full(path,filetype, onlyname=None):
    name =[]
    final_name = []
    # for root,dirs,files in os.walk(path):
    for _,_,files in os.walk(path):
        for i in files:
            if filetype in i:
                name.append(i.replace(filetype,''))#生成不带后缀的文件名组成的列表
    if onlyname == 1:
        final_name = [item[:-1]  for item in name]#生成无后缀的文件名组成的列表
    else:
        final_name = [path +'/' + item + filetype for item in name]#生成后缀的文件名组成的列表
    return final_name#输出由有后缀的文件名组成的列表

def get_name_fromPath(mapname):
    splitstr1 = '/'
    splitstr2 = '.'
    try:
        index1 = mapname.rindex(splitstr1)+1   
    except(ValueError):
        splitstr1 = '\\' # this is for paths of windows system 
        index1 = mapname.rindex(splitstr1)+1   
    index2 = mapname.index(splitstr2)
    return mapname[index1:index2]

def cut_aper(matrix,centre,rad):
    Cut=np.array(matrix[centre[1]-rad:centre[1]+rad,centre[0]-rad:centre[0]+rad])
    for m in np.arange(0,2*rad,1,int):
        for n in np.arange(0,2*rad,1,int):
            if (m-rad+0.5)**2+(n-rad+0.5)**2 > rad**2:
                Cut[m,n]=np.nan
            else:
                continue
    return Cut

def show_aper(ax,centre,rad): # ax is the axes of the current figure
    ellipse = Ellipse(xy=(centre[0],centre[1]), width = 2*rad, height = 2*rad, angle=0, edgecolor='b', fc='None', lw=2)
    ax.add_patch(ellipse)


class AstroMap(object):
    def __init__(self, name): # name is the path + name of the map
        self.name = name
    def showheader(self,num=None,hduname=None):
        raw_fits=fits.open(self.name)
        if isinstance(hduname,str):
            print(repr(raw_fits[hduname].header))
        elif isinstance(num, int):
            print(repr(raw_fits[num].header))            
        else:
            raw_fits.info() 
    def getHDU(self, hduname):
            hdu=fits.open(self.name)[hduname]
            w_p=WCS(hdu.header)
            mat_p=hdu.data
            return mat_p, w_p
    def getPara(self, parametername):
        tempStr = get_name_fromPath(self.name)
        spliter = '_'
        subStrs = tempStr.split(spliter)
        if parametername == 'Source' or parametername == 'source':
            return subStrs[0]
        elif parametername == 'Feed' or parametername == 'feed':
            return [int(i) for i in subStrs[1][5:].split('-')]
        elif parametername == 'Band' or parametername == 'band':
            return int(subStrs[2][4:])
        elif parametername == 'freq' or parametername == 'Freq':
            return iband2fre(subStrs[2][4:])
        elif parametername == 'attr' or parametername == 'Attr':
            return subStrs[-1]
        else:
            print('check the parameter name!')

def plot_map(mapobj, HDUname, sizeinput=None):
    fig=plt.figure()
    plt.style.use('science')
    mat, w = mapobj.getHDU(HDUname)
    med = np.nanmedian(mat)
    step= 2e-3
    ax = fig.add_subplot(111,projection=w)
    # temp=str(mapobj.getHDU(4)['iband']['value'][0])+'-'+str(mapobj.getHDU(4)['iband']['value'][1])+' '+mapobj.getHDU(4)['iband']['unit']
    # feed = int(mapobj.getHDU('Para')['feeds']['value'])
    # cutoff = float(mapobj.getHDU('Para')['cutoff']['value'])
    if isinstance(sizeinput,int) :
        ## gaussian filter
        lim=np.arange(-10*step+ med, 10*step + med, step/2)
        mat = gaussian_filter(mat, sizeinput)
        h = ax.contour(mat,lim,origin='lower', cmap='jet')
        # plt.title(f'feed = {feed}, cutoff = {cutoff}K' +', '+ temp + f'\nGaussian filter size = {sizeinput} pixel')
    else:
        ### no filter
        h = ax.imshow(mat,vmin=-1e-2,vmax=1e-3,origin='lower', cmap='jet')
        # plt.title(f'feed = {feed}, cutoff = {cutoff}K' +', '+ temp)

    plt.xlabel('RA')
    plt.ylabel('DEC')
    cb=plt.colorbar(h)
    cb.set_label('T/K')
    plt.show()


def plot_diffmap(mapobj1, mapobj2, centre, radius):
    fig=plt.figure()
    plt.style.use('science')
    mat1, w = mapobj1.getHDU('primary')
    mat2  = mapobj2.getHDU('primary')[0]
    mat = mat1 - mat2
    med = np.nanmedian(mat)
    step= 2e-3
    ax = fig.add_subplot(111,projection=w)
    ### no filter
    h = ax.imshow(mat,vmin=-10*step+ med,vmax=5*step + med,origin='lower', cmap='jet')
    # plt.title(f'feed = {feed}, cutoff = {cutoff}K' +', '+ temp)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    cb=plt.colorbar(h)
    cb.set_label('T/K')
    show_aper(ax,centre,radius)


def jackknife(mapobj1, centre, rad, mapobj2=None):
    pri_mat1  = cut_aper(mapobj1.getHDU('primary')[0], centre, rad)
    cov_mat1  = cut_aper(mapobj1.getHDU('covariance')[0], centre, rad)
    # hit_mat1  = cut_aper(mapobj1.getHDU('hits')[0],centre,rad)
    # integral_time = np.nansum(hit_mat1)/50   # sample rate is 50 Hz
    if isinstance(mapobj2, AstroMap):
        pri_mat2  = cut_aper(mapobj2.getHDU('primary')[0], centre, rad)
        # hit_mat2  = cut_aper(mapobj2.getHDU('hits')[0],centre,rad)
        # integral_time2 = np.nansum(hit_mat2)/50 
        # cov_mat2  = cut_aper(mapobj2.getHDU('covariance')[0],centre, rad)
        diff_mat = pri_mat1 - pri_mat2
        mean_diff = np.nanmean(diff_mat)
        std_diff = np.nanstd(diff_mat)
        # std_diff = np.nanstd(diff_mat)  # rms over intergal time
        return mean_diff, std_diff

    else:
        mean_single = np.nansum(pri_mat1/cov_mat1)/np.nansum(1/cov_mat1)
        std_single = np.sqrt(1/np.nansum(1/cov_mat1)) # this is the std calculated from the covariance map
        std_pixel = np.nanstd(pri_mat1)  # this is the std calculated from the pixel values
        return mean_single, std_single, std_pixel

def T2flux(Temperture, freq):
    # freq in units of GHz, 
    # return is the flux density in Jys
    factor = 2.59971*10**-3
    return factor*freq**2*Temperture

def photometry(mapobj, centre, a_ellipse, b_ellipse, theta, annulus_width):
    '''
    photometry function：
    centre:
    centre is a pair of numbers in units of degrees which is the loacation of objects

    a_ellipse:
    the semi-major axis in pixels

    b_ellipse:
    the semi-minor axis

    theta:
    The rotation angle in radians of the ellipse semimajor axis from the positive x axis.
    The rotation angle increases counterclockwise. The default is 0.
    '''
    distance_annu_aper = 2  # the distance between the aperture and the annulus in units of pixels
    pri_data, wcs_data = mapobj.getHDU('primary')
    Freq = np.nanmean(mapobj.getPara('freq'))
    x_pix,y_pix = wcs_data.wcs_world2pix(centre[0],centre[1],0)
    centre_pix = [(x_pix, y_pix)]

    aperture = EllipticalAperture(centre_pix, a_ellipse, b_ellipse, theta)
    annulus_aperture = EllipticalAnnulus(centre_pix, a_ellipse+distance_annu_aper, a_ellipse+annulus_width+distance_annu_aper, 
    b_ellipse+annulus_width+distance_annu_aper, theta=theta)
    annulus_masks = annulus_aperture.to_mask(method='center')
    bkg_median = []
    bkg_std =[]
    for mask in annulus_masks:
        annulus_data = mask.multiply(pri_data)
        annulus_data_1d = annulus_data[mask.data > 0]
        _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
        bkg_median.append(median_sigclip)
        bkg_std.append(np.nanstd(annulus_data_1d))
    bkg_median = np.array(bkg_median)
    bkg_std = np.array(bkg_std)
    phot_table = aperture_photometry(pri_data, aperture)
    phot_table['annulus_median'] = bkg_median
    phot_table['aper_bkg'] = bkg_median * aperture.area
    phot_table['aper_sum_bkgsub'] = phot_table['aperture_sum'] - phot_table['aper_bkg']
    phot_table['aper_sum_bkgsub_Jy'] = T2flux(phot_table['aper_sum_bkgsub'], Freq)
    phot_table['bkg_rms_Jy']= T2flux(bkg_std,Freq)

    for col in phot_table.colnames:
        phot_table[col].info.format = '%.8g'  # for consistent table output
    return phot_table

def sp_plot(source):
    fre=source['freq']
    flux=source['flux']
    flux_err=source['flux_err']
    plt.figure()
    plt.style.use('science')
    plt.errorbar(fre,flux,yerr = flux_err,fmt='o',ecolor='r',color='b',elinewidth=2,capthick=2,capsize=4,markersize=6,label='Photometry data')
    plt.xlabel('Frequency/GHz')
    plt.ylabel('Flux Density/Jy')
    # plt.title(source['name'])

def fitting_plot(source):
    index=stats_tools.s_index(source)
    temp=stats_tools.para(source)
    # fre=source['fre']
    fre=np.linspace(26,34,20)
    plt.plot(fre,np.power(fre,index[0])*np.power(e,temp[0]),'b--',linewidth=2,label='Fitting')

def T_Tplot(mapobj1, mapobj2):
    data1, wcs1 = mapobj1.getHDU('primary')
    data2, wcs2 = mapobj2.getHDU('primary')
    data1 = cut_aper(data1, [240,240],40)
    data2 = cut_aper(data2, [240,240],40)
    list1, list2 = stats_tools.pairFrom2mat(data1,data2)
    plt.figure()
    plt.plot(list1, list2,'.')
    plt.show()
    
if __name__ == '__main__':
    path = f"C:/Users/Shibo/Desktop/COMAP-sem2/week9/maps/feed3_old"

    filenames_un = get_filename_full(path, 'fits')
    onlynames = get_filename_full(path, 'fits',1)
    filenames = sortbyIband(onlynames, filenames_un)
    mapnames = [get_name_fromPath(fname) for fname in filenames]
    mapobjs = []
    for onefile in filenames:
        mapobjs.append(AstroMap(onefile))
    T_Tplot(mapobjs[0],mapobjs[5])