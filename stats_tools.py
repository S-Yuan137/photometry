from math import e
import numpy as np

# 定义计算离散点导数的函数
def cal_deriv(x, y):                  # x, y的类型均为列表
    diff_x = []                       # 用来存储x列表中的两数之差
    for i, j in zip(x[0::], x[1::]):  
        diff_x.append(j - i)
 
    diff_y = []                       # 用来存储y列表中的两数之差
    for i, j in zip(y[0::], y[1::]):
        diff_y.append(j - i)  
        
    slopes = []                       # 用来存储斜率
    for i in range(len(diff_y)):
        slopes.append(diff_y[i] / diff_x[i])
        
    deriv = []                        # 用来存储一阶导数
    for i, j in zip(slopes[0::], slopes[1::]):        
        deriv.append((0.5 * (i + j))) # 根据离散点导数的定义，计算并存储结果
    deriv.insert(0, slopes[0])        # (左)端点的导数即为与其最近点的斜率
    deriv.append(slopes[-1])          # (右)端点的导数即为与其最近点的斜率

    return deriv                      # 返回存储一阶导数结果的列表
# calculate the chi^2/degree of freedom to evaluate the fitting quality    
def chi_sqare(source):
    index=s_index(source)
    temp=para(source)
    fre=source['freq']
    flux=source['flux']
    flux_err=source['flux_err']
    flux_fitting=np.power(fre,index[0])*np.power(e,temp[0])
    chi_s=np.sum(np.power((flux_fitting-flux)/flux_err,2))
    chi_s_Ndof=chi_s/2
    return chi_s_Ndof
# Y = beta1 * X + beta0, this is beta1
# compute the spectral index and its error
def s_index(source):
    fre=source['freq']
    flux=source['flux']
    flux_err=source['flux_err']
    lnfre=np.log(fre)
    lnflux=np.log(flux)
    lnflux_err=np.abs(flux_err/flux)
    alpha_value=ordinary_least_squares(lnfre,lnflux)[0]
    alpha_err=ordinary_least_squares_err(lnfre,lnflux,lnflux_err)[0]
    alpha=np.array([alpha_value,alpha_err])
    return alpha
# Y = beta1 * X + beta0, this is beta0
def para(source):
    fre=source['freq']
    flux=source['flux']
    flux_err=source['flux_err']
    lnfre=np.log(fre)
    lnflux=np.log(flux)
    lnflux_err=np.abs(flux_err/flux)
    para_value=ordinary_least_squares(lnfre,lnflux)[1]
    para_err=ordinary_least_squares_err(lnfre,lnflux,lnflux_err)[1]
    para=np.array([para_value,para_err])
    return para
#  Y = beta1 * X + beta0
#  Y has errors: Y_err
def ordinary_least_squares(X,Y):  
    X_mean=np.mean(X)
    Y_mean=np.mean(Y)
    beta1=np.sum((X-X_mean)*(Y-Y_mean))/np.sum(np.power(X-X_mean,2))
    beta0=Y_mean-beta1*X_mean
    return beta1,beta0
# calculate the err of least square fitting 
def ordinary_least_squares_err(X,Y,Y_err):
    X_mean=np.mean(X)
    # Y_mean=np.mean(Y)
    Y_mean_rms=1/len(Y_err)*np.sqrt(np.sum(np.power(Y_err,2)))
    # beta1_rms=1/A*sqrt(B)
    A=np.sum(np.power(X-X_mean,2))
    B=np.sum(np.power((X-X_mean)*Y_err,2))
    beta1_rms=1/A*np.sqrt(B)
    beta0_rms=np.sqrt(np.power(Y_mean_rms,2)+np.power(beta1_rms,2))
    return beta1_rms, beta0_rms

def pairFrom2mat(mat1, mat2):
    '''
    generate the pairs of numbers at the same position in two matrices 
    '''
    number1 = []
    number2 = []
    if mat1.shape == mat2.shape:
        for i in np.arange(0,mat1.shape[0],1,int):
            for j in np.arange(0,mat1.shape[1],1,int):
                number1.append(mat1[i,j])
                number2.append(mat2[i,j])
        return np.array(number1), np.array(number2)
    else:
        # to be continue
        return 'to be continue'

if __name__ == '__main__':
    # temp test
    mat1 = np.array([[1,2,3],[4,9,6]])
    print(pairFrom2mat(mat1, mat1))