import numpy as np
from matplotlib.patches import Ellipse
import os 
from astropy.io import fits                                  #package for read/write/save fits files
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter


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
    list1_s, list2_s= [list(x) for x in result]
    return list2_s

def sortbyIband(names, filenames): # defalut format is the last character of names is the iband number
    iband = [iname[-1] for iname in names]
    fre_start = [iband2fre(i)[0] for i in iband]
    return sort_list(fre_start, filenames)

def get_filename_full(path,filetype, onlyname=None):
    name =[]
    final_name = []
    for root,dirs,files in os.walk(path):
        for i in files:
            if filetype in i:
                name.append(i.replace(filetype,''))#生成不带后缀的文件名组成的列表
    if onlyname == 1:
        final_name = [item[:-1]  for item in name]#生成无后缀的文件名组成的列表
    else:
        final_name = [path +'/' + item + filetype for item in name]#生成后缀的文件名组成的列表
    return final_name#输出由有后缀的文件名组成的列表

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
        # this if part is for the parameters HDU, maybe changed in the future
        if hduname == 4 or hduname== 'Para': 
            hdu=fits.open(self.name)[hduname]
            # w_p=WCS(hdu.header)
            para_dict = {}
            table_p = hdu.data
            para_names = table_p.columns.names
            para_units = table_p.columns.units
            for num,para in enumerate(para_names):
                if para =='iband': # convert iband to frequency
                    para_dict.update({para : {'value': iband2fre(int(table_p[para])), 'unit' : 'GHz'}})
                else:
                    para_dict.update({para : {'value': table_p[para], 'unit' : para_units[num]}})
            return para_dict

        else:
            hdu=fits.open(self.name)[hduname]
            w_p=WCS(hdu.header)
            mat_p=hdu.data
            return mat_p, w_p
