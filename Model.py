#!/usr/bin/python3
import numpy as np
from scipy.stats import matrix_normal, invgamma, gamma, truncnorm
import time
# from trun_mvnt import rtmvn3
import pandas as pd
import cv2
import matplotlib.image as mpimg

def sample_tmn(mean, rowcov, colcov):
    num_f = -1
    
    tic = time.perf_counter()
    kron = np.kron(colcov, rowcov)
    vecv = [0]
    while np.sum(vecv)==0:
        num_f += 1
        vecv = np.array(rtmvn(Mean=mean.flatten(order='F'), Sigma=kron, burn=5, thin=1))
        if num_f >=1:
            print('Fail to sampling from TMN. Num_fail =', num_f, end='\r')
        if num_f == 20:
            print('Fail to sampling from TMN, use RMN instead.')
            vecv = matrix_normal.rvs(mean=mean, rowcov=rowcov, colcov=colcov)
            vecv[vecv<0]=0
            break

    v = vecv.reshape((mean.shape[1], mean.shape[0])).T
    toc = time.perf_counter()
    print('Time of sampling one v =', toc - tic, 'Average time =', (toc - tic)/6)
    return v


#' Random number generation for truncated multivariate normal distribution subject to nonnegativity constraints
def rtmvn(Mean, Sigma, burn=5, thin=100):
    # simplified version
    p = len(Mean)
    Sigma = np.matrix(Sigma)
    M = np.matrix(Mean).reshape((p, 1))

    ini = M.copy()
    ini[ini<=0] = 0.01 # np.min(np.abs(M))

    R = np.linalg.cholesky(Sigma)
    z = np.linalg.solve(R, ini - M) # initial value
    Rz = np.matmul(R, z)
    out = []

    # for i in range((thin + 1) * n + burn):
    for i in range(thin + burn):
        tic = time.perf_counter()
        for j in range(p):
            z_oldj = z[j].copy()
            rj = R[:, j] 
            a_temp = -M-(Rz-rj*z[j]) # replace a_temp = -M-np.matmul(Rj, zj)

            # ignoring rj = 0, as alwyays fulfill
            pos = np.array(rj > 0)
            neg = np.array(rj < 0)

            if pos.sum() == 0: 
                lower_j = -np.inf
            else: 
                lower_j = (a_temp[pos] / rj[pos]).max()  # when r_jk>0

            if neg.sum() == 0:
                upper_j = np.inf
            else:
                upper_j = (a_temp[neg] / rj[neg]).min()  # when r_jk<0,

            if lower_j>=upper_j:
                # print('Fail to sampling from TMN! One more sample needed.')
                return 0

            z[j] = truncnorm.rvs(a=lower_j, b=upper_j, loc=0, scale=1, size=1)
            Rz += rj*(z[j] - z_oldj)
        # forEnd

        if i>=burn:
            out.append( (np.matmul(R, z) + M).reshape(1, p) )

        toc = time.perf_counter()
        # print('time of samplingn = ', toc - tic)
    # forEnd
    return np.mean(out, axis=0)


def basin(x0, b0, L, burn, thin, tmn=True, save = None):

    n_loc = x0.shape[1]
    n_type = b0.shape[1]
    n_gene = x0.shape[0]
    lam = 0.0
    if n_loc*n_type>=10000:
        print('Data is too large, use RMN.')
        tmn=False

    Ip = np.identity(n_loc)
    Ic = np.identity(n_type)
    invL = np.linalg.inv(L+0.01*Ip)
    btb = np.matmul(b0.T, b0)
    btx = np.matmul(b0.T, x0)


    # initialize
    v = np.random.uniform(0, 1, size=(n_type, n_loc))
    # v = np.ones((n_type, n_loc))
    Iv = np.ones_like(v)
    IvinvL = np.matmul(Iv, invL)
    result = np.zeros([thin, n_type, n_loc], dtype=float)
    for i in range(n_loc):
        v[:, i] = v[:, i] / np.sum(v[:, i])

    # start sampling
    for i in range(burn + thin):

        # sample sigma^2
        E = x0 - np.matmul(b0, v)
        temp1_sigma = np.matmul( np.matmul(L, E.T), E)
        temp2_sigma = np.matmul( np.matmul(L, v.T), v)
        sigma= invgamma.rvs(a= (n_gene*n_loc + n_type*n_loc)*0.5, loc=0,
                           scale= 0.5*( np.trace(temp1_sigma) + lam*np.trace(temp2_sigma)), size=1)

        # sample eta
        temp_eta = np.trace(np.matmul(Iv.T, v))
        eta = gamma.rvs( a= n_type*n_loc, loc=0, scale = 1/(temp_eta) )


        # sample V
        cov = np.linalg.pinv( btb + lam*Ic )
        mean = np.matmul( cov, btx - sigma*eta*IvinvL )
        if tmn==True:
            v = sample_tmn(mean, sigma * cov, invL)
        else:
            v = matrix_normal.rvs(mean=mean, rowcov=sigma*cov, colcov=invL)
            v[v<0]=0

        # normalize v
        for k in range(n_loc):
            v[:, k] = v[:, k] / (np.sum(v[:, k]) + 1e-8)

        # save v into 3D matrix
        if i >= burn:
            result[i-burn, :, :] = v
            if save != None:
                path = save + '\\out_' + str(i-burn) + '.csv'
                np.savetxt(path, v.T, delimiter=",")

        print('Sample #:', i, 'min=', np.min(v), 'max=', np.max(v),)
    # loop end

    # normalize and compute mean and std
    v_mean = np.mean(result, axis=0)
    v_std = np.std(result, axis=0)
    for i in range(n_loc):
        v_mean[:, i] = v_mean[:, i] / (np.sum(v_mean[:, i]) + 1e-6)

    return v_mean, v_std


def run_basin(st_selected, st_loc, sc_ave, img_path=None, sigma=0.1, burn=5, thin=100, tmn=True, save=None):

    # image parameters

    # keys, keynum = np.unique(sc_meta['CellType'], return_counts=True)
    keys = list(sc_ave.columns)
    keys.remove('Genes')
    x0 = np.array( st_selected.drop( labels='Genes', axis=1) ,dtype=float )
    b0 = np.array(sc_ave.drop( labels='Genes', axis=1) ,dtype=float )

    # normalize ST data
    x0 = np.log( 1 + x0/np.sum(x0) )
    x0 = x0/np.mean(x0)

    # get the locations
    n_loc = x0.shape[1]
    loc = np.zeros((n_loc, 2))
    loc[:, 0] = st_loc['x']
    loc[:, 1] = st_loc['y']


    # compute graph laplacian and adjacency matrix
    mu_intensity = 1.0
    mu = 1.0
    crop = 2
    isize = 1024
    A = np.zeros((n_loc,n_loc))
    D = np.zeros((n_loc,n_loc))
    if img_path==None:
        for i in range(n_loc):
            for j in range(n_loc):
                distance = (loc[i, 0]-loc[j, 0])**2 + (loc[i, 1]-loc[j, 1])**2
                A[i, j] = np.exp(-mu * distance / sigma)
    else:
        img0 = mpimg.imread(img_path)
        img = cv2.resize(img0, (isize, isize))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # img = clahe.apply(img)
        img = img/np.max(img)
        xalign = np.array(st_loc['x_align']*isize/img0.shape[1], dtype=int)
        yalign = np.array(st_loc['y_align']*isize/img0.shape[0], dtype=int)
        for i in range(n_loc):
            for j in range(n_loc):
                distance = (loc[i, 0]-loc[j, 0])**2 + (loc[i, 1]-loc[j, 1])**2
                intensity_diff = img[xalign[i]-crop:xalign[i]+crop, yalign[i]-crop:yalign[i]+crop] - img[xalign[j]-crop:xalign[j]+crop, yalign[j]-crop:yalign[j]+crop]
                # intensity_dis = (np.mean( np.mean(intensity_diff, axis=0), axis=0))**2
                intensity_dis = np.linalg.norm(intensity_diff, 2)**2
                A[i, j] = np.exp(-( mu * distance + mu_intensity * intensity_dis ) / sigma)

    for i in range(n_loc):
        A[i,i] = 0
        D[i,i] = np.sum(A[i,:])
    L = D-A
 


    mean, std = basin(x0, b0, L=L, burn=burn, thin=thin, tmn=tmn, save=save)

    # save results, create data frame
    mean_data = pd.DataFrame(data=mean.T, columns=keys)
    std_data = pd.DataFrame(data=std.T, columns=keys)

    return mean_data, std_data