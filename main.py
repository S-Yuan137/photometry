import sy_class
import matplotlib.pyplot as plt

# def map_plot(fig, ax,map,w,n):
#     # fig = plt.figure()
#     # fig.add_subplot(111,projection=w)
    
#     if n==0:
#         vminval=-1*1e-2
#         vmaxval=1*1e-2
#         label='T/K'
#         title='Primary'
#     elif n==1:
#         vminval=0*1e-6
#         vmaxval=5*1e-7
#         label='Undefine'
#         title='covariance'
#     elif n==2:
#         vminval=-1
#         vmaxval=1  
#         label='Undefine'
#         title='Hit'
#     elif n==3:
#         vminval=-1*1e-2
#         vmaxval=1*1e-2 
#         label='Undefine'
#         title='naive'
                   
#     h=ax.imshow(map,vmin=vminval,vmax=vmaxval,origin='lower', cmap='jet')
#     plt.xlabel('RA')
#     plt.ylabel('DEC')
#     # plt.title(title)
#     cb=plt.colorbar(h)
#     cb.set_label(label)

path = r'C:\Users\Shibo\Desktop\COMAP-sem2\maps\maps4e_3'
filenames = sy_class.get_filename(path, 'fits')

h= plt.figure()
plt.style.use('science')
for num, obj in enumerate(filenames):
    map1 = sy_class.AstroMap(path+'/'+obj)
    mat, w = map1.getHDU('primary')
    ax = h.add_subplot(2, 4, num+1,projection=w)
    # ax = h.add_subplot(2, 4, num+1)
    # map_plot(h, ax, mat, w, 0)
    subplot=ax.imshow(mat,vmin=-1e-2,vmax=1e-2,origin='lower', cmap='jet')
   
    plt.xlabel('RA')
    plt.ylabel('DEC')
    temp=str(map1.getHDU(4)['iband']['value'][0])+'~'+str(map1.getHDU(4)['iband']['value'][1])+' '+map1.getHDU(4)['iband']['unit']
    plt.title(temp)
    
plt.tight_layout()
cb=plt.colorbar(subplot)
cb.set_label('T/K')
plt.show()