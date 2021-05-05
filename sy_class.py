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
import math
import re
import sys
import scipy.ndimage                                        # for interpolate the data (smooth)


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


def sort_list(list1,list2): #according to list1 to sort two lists
    try:      # try convert the list1 elements to numbers
        list1 = [float(num) for num in list1]
        sortIndex = np.argsort(list1)
        return [list2[i] for i in sortIndex]
    except:   # avoid using np.argsort if the input arguments are lists(not numbers), even if different length
        zipped=zip(list1,list2)
        sort_zipped = sorted(zipped,key=lambda x:(x[0]))
        result = zip(*sort_zipped) # 将 sort_zipped 拆分成两个元组
        # list1_s, list2_s= [list(x) for x in result]
        list2_s= [list(x) for x in result][1]
        return list2_s

    
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
    index2 = mapname.rfind(splitstr2)
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

def show_aper(ax,centre,size, theta_deg): # ax is the axes of the current figure
    ellipse = Ellipse(xy=(centre[0],centre[1]), width = 2*size[0], height = 2*size[1], angle=theta_deg, edgecolor='b', fc='None', lw=2)
    ax.add_patch(ellipse)

class AstroMap(object):
    def __init__(self, name): # name is the path + name of the map
        self.name = name
    def showheader(self,hduname=None):
        raw_fits=fits.open(self.name)
        if hduname is not None:
            try:
                print(repr(raw_fits[hduname].header))
            except:
                print('wrong hduname input')
                raw_fits.info()
        else:
            raw_fits.info()                   
    def getHDU(self, hduname = None):
        if  hduname is None:
            hduname = 0
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
        elif parametername == 'attrVal' or parametername == 'AttrVal':
            return re.findall(r"\-?\d+\.?\d*",subStrs[-1])[0]
        elif parametername == 'name':
            return tempStr
        else:
            print('check the parameter name!')
    def showmap(self,hduname=None, fig = None, subPlot = None):
        mat_data, wcs_data = self.getHDU(hduname)
        plt.style.use('science')
        if subPlot is not None and fig is not None:
            ax = fig.add_subplot(subPlot[0],subPlot[1],subPlot[2], projection = wcs_data)
        else:
            ax = plt.subplot(projection=wcs_data)
        # ax.imshow(mat_data, origin='lower', vmin=np.nanmedian(mat_data)-np.nanstd(mat_data)/10, 
        #                                      vmax=np.nanmedian(mat_data)+np.nanstd(mat_data)/10,
        #                                      cmap='jet')
        ax.imshow(mat_data, origin='lower',cmap='jet')
        ax.coords['ra'].set_axislabel('Right Ascension')
        ax.coords['dec'].set_axislabel('Declination')
        try:
            ax.set_title(str(self.getPara('freq')[0])+'-'+str(self.getPara('freq')[1])+' GHz'+ ' '+ self.getPara('attr'))
        except:
            print('Fail to get freqency!')
        return ax
        ##### in the future, add Gaussian filter and contour plot
        # some initial codes:
        # lim=np.arange(-10*step+ med, 10*step + med, step/2)
        # mat = gaussian_filter(mat, sizeinput)
        # h = ax.contour(mat,lim,origin='lower', cmap='jet')
    def showaper(self, centre_world, size, theta_deg, filtersigma, refcontour = None, hduname = None):
        ##### show the whole map and sliced map ( based on input aperture)
        mat_data, wcs_data = self.getHDU(hduname)
        x0, y0 = wcs_data.wcs_world2pix(centre_world[0], centre_world[1], 0)
        fig = plt.figure()
        ax = self.showmap(hduname, fig, [1,2,1])
        show_aper(ax, [x0, y0], size, theta_deg)
        ###### the second sliced map #######
        theta = theta_deg * (np.pi/180)
        aperture = EllipticalAperture([(x0,y0)], size[0], size[1], theta)
        aperture_mask = aperture.to_mask(method= 'center')[0]
        aper_mat = aperture_mask.multiply(mat_data)
        y_Npix, x_Npix = aper_mat.shape
        x1 = round(x0-x_Npix/2)
        x2 = round(x0+x_Npix/2)
        y1 = round(y0-y_Npix/2)
        y2 = round(y0+y_Npix/2)
        wcs_cut = wcs_data[y1:y2, x1:x2]
        mat_cut = mat_data[y1:y2, x1:x2]
        mat_cut = gaussian_filter(mat_cut, sigma= filtersigma)
        # these two are for future usage
        # vmin = np.nanmedian(mat_data)-np.nanstd(mat_data)/10 
        # vmax = np.nanmedian(mat_data)+np.nanstd(mat_data)/10
        ax2 = fig.add_subplot(122, projection = wcs_cut)
        ax2.imshow(mat_cut,origin='lower', cmap='jet')
        # contour has to be sliced as well
        # colors = 
        
        if refcontour is not None:
            ref_mat, ref_wcs = refcontour.getHDU(hduname)
            ref_x0, ref_y0 = ref_wcs.wcs_world2pix(centre_world[0], centre_world[1], 0)
            aperture = EllipticalAperture([(ref_x0,ref_y0)], size[0], size[1], theta)
            aperture_mask = aperture.to_mask(method= 'center')[0]
            aper_mat = aperture_mask.multiply(ref_mat)
            y_Npix, x_Npix = aper_mat.shape
            x1 = round(x0-x_Npix/2)
            x2 = round(x0+x_Npix/2)
            y1 = round(y0-y_Npix/2)
            y2 = round(y0+y_Npix/2)
            wcs_cut = ref_wcs[y1:y2, x1:x2]
            mat_cut = ref_mat[y1:y2, x1:x2]
            mat_cut = gaussian_filter(mat_cut, sigma= filtersigma)
            levels = np.linspace(np.nanmin(mat_cut), np.nanmax(mat_cut), 10)
            print(f'contour levels in the aperSubplot:{levels}') # show the contour levels
            ax2.contour(np.arange(mat_cut.shape[1]), np.arange(mat_cut.shape[0]), mat_cut, colors= 'k', levels=levels,
                    linewidths=1, smooth=16)
        else:
            levels = np.linspace(np.nanmin(mat_cut), np.nanmax(mat_cut), 5)
            print(f'contour levels in the aperSubplot:{levels}') # show the contour levels
            ax2.contour(np.arange(mat_cut.shape[1]), np.arange(mat_cut.shape[0]), mat_cut, colors= 'k', levels=levels,
                    linewidths=1, smooth=16)
        ax2.coords['ra'].set_axislabel('Right Ascension')
        ax2.coords['dec'].set_axislabel('Declination')

def sortMaps(mapobjList, sortIndicator):
    indicators = []
    for mapobj in mapobjList:
        indicators.append(mapobj.getPara(sortIndicator))
    return sort_list(indicators, mapobjList)

def getMapList(mapFolder, sortIndicator = None):
    mapdirs = get_filename_full(mapFolder, 'fits')
    mapobjs = []
    for onefile in mapdirs:
        mapobjs.append(AstroMap(onefile))
    if sortIndicator is not None:
        return sortMaps(mapobjs, sortIndicator)
    else:
        return mapobjs

def plot_map(mapobjList, HDUname):
    fig=plt.figure()
    plt.style.use('science')
    Num = len(mapobjList)
    x_num = math.floor(np.sqrt(Num))
    y_num = math.ceil(Num/x_num)
    for No, obj in enumerate(mapobjList):
        subPlot = [int(x_num), int(y_num), int(No+1)] # the location of the subplot
        obj.showmap(HDUname, fig, subPlot)

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
    # h = ax.imshow(mat,origin='lower', cmap='jet')
    # plt.title(f'feed = {feed}, cutoff = {cutoff}K' +', '+ temp)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    cb=plt.colorbar(h)
    cb.set_label('T/K')

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
        std_single = np.sqrt(np.nansum(cov_mat1)) # this is the std calculated from the covariance map
        std_pixel = np.nanstd(pri_mat1)  # this is the std calculated from the pixel values
        return mean_single, std_single, std_pixel

def T2flux(Temperture, freq):
    # freq in units of GHz, 
    # return is the flux density in Jys
    factor = 2.59971*10**-3
    return factor*freq**2*Temperture

def photometry(mapobj, centre, a_ellipse, b_ellipse, theta_deg, annulus_width):
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
    distance_annu_aper = 5  # the distance between the aperture and the annulus in units of pixels
    theta = theta_deg * (np.pi/180)
    pri_data, wcs_data = mapobj.getHDU('primary')
    cov_data,_ = mapobj.getHDU('covariance')
    
    Freq = np.nanmean(mapobj.getPara('freq'))
    x_pix,y_pix = wcs_data.wcs_world2pix(centre[0],centre[1],0)
    centre_pix = [(x_pix, y_pix)]

    aperture = EllipticalAperture(centre_pix, a_ellipse, b_ellipse, theta)
    annulus_aperture = EllipticalAnnulus(centre_pix, a_ellipse+distance_annu_aper, a_ellipse+annulus_width+distance_annu_aper, 
    b_ellipse+annulus_width+distance_annu_aper, theta=theta)
    temp_mask = aperture.to_mask(method= 'center')
    annulus_masks = annulus_aperture.to_mask(method='center')
    bkg_median = []
    bkg_std =[]
    for mask in annulus_masks:
        annulus_data = mask.multiply(pri_data)
        annulus_data_1d = annulus_data[mask.data > 0]
        _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
        bkg_median.append(median_sigclip)
        # bkg_std.append(np.nanstd(annulus_data_1d))
    bkg_median = np.array(bkg_median)
    for t_mask in temp_mask:
        cov_mat1 = t_mask.multiply(cov_data)[t_mask.data>0]
        bkg_std.append(np.sqrt(np.nansum(cov_mat1)))
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

def T_Tplot(mapobj1, mapobj2, centre_world, size, theta_deg, downsample = [None, None], filtersigma = None ):
    '''
    the downsample parameter should match the size of the aperture, generally 1/10 of the size

    filtersigma is the sigma of the Gaussian filter sigma
    '''
    data1, wcs_data = mapobj1.getHDU('primary')
    x0, y0 = wcs_data.wcs_world2pix(centre_world[0], centre_world[1], 0)
    if filtersigma is not None:
        data1 = gaussian_filter(data1, sigma= filtersigma)

    plt.figure()
    plt.style.use('science')
    for map2 in mapobj2:
        data2, _ = map2.getHDU('primary')
        if filtersigma is not None:
            data2 = gaussian_filter(data2, sigma= filtersigma)
        list1, list2 = stats_tools.pairFrom2mat(data1,data2, [x0, y0], size, theta_deg, downsample)
        plt.plot(list1, list2,'.', label = map2.getPara('attr'))
        print(len(list1))

    plt.xlabel('Temperature values at 4.85 GHz')
    plt.ylabel('Temperature values at 26.5 GHz')
    if len(mapobj2)>1:
        plt.legend()
    plt.show()

def NoiseStd(mapobjList, centre_world, size, theta_deg, downsample = [None, None], Covariance =None):
    '''
    estimate the background std from two aspects: covariance and pixel values
    
    maps in mapobjList should have same wcs header

    default aperture should be (bottom left )
    centre_world = [12.7452212, 40.4682106]
    size = [17.00, 5.00]
    theta_deg = 0

    '''
    bkgSTD_pix = []                 # this is the std calculated from the pixel values
    if Covariance is not None:
        bkgSTD_cov = []                 # this is the std calculated from the covariance map
        for mapobj in mapobjList:
            pri_mat, wcs_header = mapobj.getHDU('primary')
            cov_mat, _ = mapobj.getHDU('covariance')
            x0, y0 = wcs_header.wcs_world2pix(centre_world[0], centre_world[1], 0)
            pri_mat = stats_tools.maskCut(pri_mat, [x0, y0], size, theta_deg, downsample)
            cov_mat = stats_tools.maskCut(cov_mat, [x0, y0], size, theta_deg, downsample)
            bkgSTD_cov.append(np.sqrt(np.nansum(cov_mat)))
            bkgSTD_pix.append(np.nanstd(pri_mat)) 
        return bkgSTD_pix, bkgSTD_cov
    else:
        for mapobj in mapobjList:
            pri_mat, wcs_header = mapobj.getHDU('primary')
            x0, y0 = wcs_header.wcs_world2pix(centre_world[0], centre_world[1], 0)
            pri_mat = stats_tools.maskCut(pri_mat, [x0, y0], size, theta_deg, downsample)
            bkgSTD_pix.append(np.nanstd(pri_mat)) 
        return bkgSTD_pix


def WeightAverageMap(mapobjList, outputName, outputDir):
    '''
    use multiple maps to generate a weight averaged maps from each feed
    '''
    fout = outputDir + '/' + outputName
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    elif os.path.exists(fout):
        os.remove(fout)

    pri_matList = []
    wei_matList = []
    wcs_header = mapobjList[0].getHDU('primary')[1]
    for mapobj in mapobjList:
        pri_matList.append(mapobj.getHDU('primary')[0])
        wei_matList.append(1/mapobj.getHDU('covariance')[0])
    final_mat = stats_tools.AddMatrices(pri_matList, wei_matList)
    grey=fits.PrimaryHDU(final_mat, header= wcs_header.to_header())
    greyHDU=fits.HDUList([grey])
    greyHDU.writeto(fout)

def WeightAverageMapVersion2(mapobjList, outputName, outputDir):
    fout = outputDir + '/' + outputName
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    elif os.path.exists(fout):
        os.remove(fout)

    pri_matList = []
    wei_matList = []
    wcs_header = mapobjList[0].getHDU('primary')[1]
    for mapobj in mapobjList:
        pri_matList.append(mapobj.getHDU('primary')[0])
        tempNum = int(mapobj.getPara('attrVal'))
        wei_matList.append(tempNum/mapobj.getHDU('covariance')[0])
    final_mat = stats_tools.AddMatrices(pri_matList, wei_matList)
    grey=fits.PrimaryHDU(final_mat, header= wcs_header.to_header())
    greyHDU=fits.HDUList([grey])
    greyHDU.writeto(fout)


if __name__ == '__main__':
    Ref = sys.argv[1]
    PC  = sys.argv[2]
    refmap = AstroMap('C:/Users/Shibo/Desktop/COMAP-sem2/week11/m31cm6i_3min_ss_on_fg4.fits')
    mapobj2 = AstroMap(f'C:/Users/Shibo/Desktop/COMAP-sem2/week13/AddFeedsMaps/Ref{Ref}_FeedsAll_Band0_PC{PC}.fits')
    # mapobj2v2 = AstroMap(f'C:/Users/Shibo/Desktop/COMAP-sem2/week13/AddFeedsMaps/AddFeedsMapsVersion2/Ref{Ref}_FeedsAll_Band0_PC{PC}.fits')
    
    # mapobj = AstroMap('C:/Users/Shibo/Desktop/COMAP-sem2/week10/maps/fg4_Feeds1-2-3-5-6-8-9-10-11-12-13-14-15-16-17-18-19_Band0.fits')
    NoiseAper = {'centre':np.array([12.7452212, 40.4682106]), 'size': np.array([17, 5]), 'theta' : 0}
    M31 ={'centre':np.array([10.6836, 41.2790]), 'size':np.array([60,20]), 'theta':127}
    M31part = {'centre':np.array([11.0512218, 41.3032980]), 'size':np.array([30,15]), 'theta':120}
    # RG5C3_50 = {'centre':np.array([9.6076856,41.6096426]), 'size':np.array([6,6]), 'theta':0}
    # # print(jackknife(mapobj2, [240,240], 40))
    # # mapobj2.showaper(M31['centre'], M31['size'],M31['theta'])
    # T_Tplot(refmap, [mapobj2], M31part['centre'], M31part['size'], M31part['theta'],[4,4])
    # # T_Tplot(mapobj1, [mapobj2], RG5C3_50['centre'], RG5C3_50['size'], RG5C3_50['theta'])
    # # mapobj2.showaper(RG5C3_50['centre'], RG5C3_50['size'], RG5C3_50['theta'])
    mapobj2.showaper(M31part['centre'], M31part['size'], M31part['theta'], 1.2, refmap)
    # refmap.showaper(M31['centre'], M31['size'], M31['theta'], 1.2, refmap)
    T_Tplot(refmap, [mapobj2], M31part['centre'], M31part['size'], M31part['theta'], [3,3], 1.2)
    plt.show()
    # print(NoiseStd([mapobj2, mapobj2v2], NoiseAper['centre'], NoiseAper['size'], NoiseAper['theta']))
    
    
    ################ average maps ####################################################################
    '''
    feed = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19]
    maplist = []
    Ref = sys.argv[1]
    PC = sys.argv[2]

    for i in feed:
        path = f'C:/Users/Shibo/Desktop/COMAP-sem2/week13/maps_sig_cuts_ref{Ref}/feed{i}_band0'
        # path = f'C:/Users/Shibo/Desktop/COMAP-sem2/week13/maps_sig_cuts_ref10/feed{i}_band0'
        maplist.extend(getMapList(path, 'attrVal'))
        # maplist.append(AstroMap(f'C:/Users/Shibo/Desktop/COMAP-sem2/week13/maps_sig_cuts_ref{Ref}/feed{i}_band0/fg4_Feeds{i}_Band0_PC{PC}.fits'))
    # WeightAverageMap(maplist, f'Ref{Ref}_FeedsAll_Band0_PC{PC}.fits', 'C:/Users/Shibo/Desktop/COMAP-sem2/week13/AddFeedsMaps')
    WeightAverageMapVersion2(maplist, f'Ref{Ref}_FeedsAll_Band0_PCAll.fits', 'C:/Users/Shibo/Desktop/COMAP-sem2/week13/AddFeedsMaps/AddFeedsMapsVersion2')
    '''
    

    
    