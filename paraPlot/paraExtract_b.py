import h5py
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
paths = r'/local/scratch/ztian/full_list/FileLists/fg4_level2.list'

path_of_files = []
time = []
obserID = []
wnoise =[]
f_R =[]
alpha = []
wnoise_mean = []
T_sys = []

#open and read the files                                                                            
with open(paths) as file_to_read:
    for lines in file_to_read.readlines():
        path_of_files.append(lines.strip('\n'))

#select the time from file names                                                                    
for i in range(0,len(path_of_files)):
    time.append(str(path_of_files[i])[60:77:])

#select the observation ID                                                                          
for i in range(0,len(path_of_files)):
    obserID.append(str(path_of_files[i])[52:59:])


#obtain the julian date from time                                                                   
julianT = []
for i in range(0,len(time)):
    julianT.append(Time(time[i].strip()[0:10]+'T'+time[i].strip()[11:13]+':'+time[i].strip()[13:15]+':'+time[i].strip()[15:17]).mjd)


#navigate to the objective files [feeds, sidebands, scan, parameters]                               
for f in path_of_files:
    h = h5py.File(f,'r')
    wnoise_mean.append(h['level2/Statistics/fnoise_fits'][0,0,0,0])
    f_R.append(h['level2/Statistics/fnoise_fits'][0,0,0,1])
    alpha.append(h['level2/Statistics/fnoise_fits'][0,0,0,2])
    #obtain the filted t_sys of each observation                                                    
    for i in range(0,64):
        temp1 = 0
        for j in range(0,h['level2/Statistics/wnoise_auto'].shape[3]):
            temp1 = temp1 + h['level2/Statistics/wnoise_auto'][0,0,i,j,0]
        wnoise.append(temp1/h['level2/Statistics/wnoise_auto'].shape[3])
    #obtain the flagged wnoise
    #obtain the system temperature                                                                  
    t_sys = [i * np.sqrt(0.02*10e9/64) for i in wnoise if i != 0]
    T_sys.append(np.median(t_sys))

#obtain sigma_1/f                                                                                   
sigma_1_f = []
for i in range(0,len(wnoise_mean)):
    sigma_1_f.append(wnoise_mean[i]**2 * (1. + (1./f_R[i])**alpha[i]))


#set the x ticks                                                                                    
x_scale = [i*(julianT[-1]-julianT[0])/5+julianT[0] for i in range(0,6)]
x_index = [Time(t,format='mjd').fits for t in x_scale]

#get the averaged T_sys                                                                             
t_sys_mean = [i * np.sqrt(0.02*10e9) for i in wnoise_mean]


