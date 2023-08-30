import numpy as np
import os, re
import time
from sys import path as syspath
from os import path as ospath
from scipy import stats
# from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import pairwise_kernels
# from sklearn.metrics.pairwise import additive_chi2_kernel
# from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from fastmle_res_jax_paral import getfullComponent as getfullComponent1
from fastmle_res_jax_paral import getRLComponent as getfullComponent2
from fastmle_res_jax_paral import getfullComponentPerm
from fastmle_res_jax_paral import getmleComponent 
from sklearn.impute import SimpleImputer
from utils import QMC_RFF

def direct(geno_matrix_in):
    N=geno_matrix_in.shape[0]
    M=geno_matrix_in.shape[1]
    exact = np.zeros((N, int((M*(M+1))/2)))
    s = 0
    for i in range(M):
        for j in range(i, M):
            feature = geno_matrix_in[:,i]*geno_matrix_in[:,j]
            if j != i:
                feature *= np.sqrt(2)
            exact[:,s] = feature
            s += 1
    exact_standard = stats.zscore(exact)

    # remove nan columns
    col_mean = np.nanmean(exact_standard, axis=0)
    inds = np.where(np.isnan(exact_standard))
    exact_standard[inds] = np.take(col_mean, inds[1])
    #exact_standard = exact_standard[:,~np.all(np.isnan(exact_standard), axis=0)]

    return exact_standard

def direct_noself(geno_matrix_in):
    N=geno_matrix_in.shape[0]
    M=geno_matrix_in.shape[1]
    exact = np.zeros((N, int((M*(M-1))/2)))
    s = 0
    for i in range(M):
        for j in range(i+1, M):
            feature = geno_matrix_in[:,i]*geno_matrix_in[:,j]
            if j != i:
                feature *= np.sqrt(2)
            exact[:,s] = feature
            s += 1
    exact_standard = stats.zscore(exact)

    # remove nan columns
    col_mean = np.nanmean(exact_standard, axis=0)
    inds = np.where(np.isnan(exact_standard))
    exact_standard[inds] = np.take(col_mean, inds[1])
    #exact_standard = exact_standard[:,~np.all(np.isnan(exact_standard), axis=0)]

    return exact_standard

def estimateSigmasGeneral(y, Xc, X, params=None, how='rand_mom',Random_state=1,method='Perm'):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    
    if how == 'mle':
        gamma = params['gamma'] if params['gamma'] is not None else (1/ (X.shape[1] * X.var()))
        Kactual = pairwise_kernels(X, metric=params['kernel_metric'], gamma=gamma)
        t0 = time.time()
        center=params['center']
        h = getmleComponent(Xc, Kactual, y, center=center)
        t1 = time.time()
        Kernel_Time = t1-t0
        states = [H[1] for H in h]
        pvals = [H[0] for H in h]
        print(f'mle pvals are: {pvals}')
        # print(f'get components takes {t1-t0}')
        return (pvals,Kernel_Time,states)
 
    elif how == 'fast_mle':
        gamma = params['gamma'] if params['gamma'] is not None else (1/ (X.shape[1]*X.var()))
        # print( gamma)
        t0 = time.time()
        center=params['center']
        # if params['version'] == 1:
        #     rbfs = RBFSampler(gamma=gamma, n_components=params['D'],random_state=Random_state)
        #     Z = rbfs.fit_transform(X)
        # elif params['version'] == 2:
        #     Z = RFF_fit_transform(X,params['D'],gamma,seed=Random_state)
        QMC=params['version']
        Test=params['Test']
        # print(QMC)
        # t0 = time.time()
        if QMC=='Vanilla':
            # rbfs = RBFSampler(gamma=gamma, n_components=params['D'],random_state=Random_state)
            # ps = PolynomialCountSketch(degree=2, coef0=0, gamma=gamma, n_components=params['D'], random_state=Random_state)
            Z = direct_noself(X)
        else:
            Z = QMC_RFF(gamma=gamma,d=X.shape[1],n_components=params['D'],seed=Random_state,QMC=QMC).fit_transform(X)
        
        print(f'Test version is {Test}')   
        if Test=='general': #commented out in Sep 4th
            Z = np.concatenate((X,Z),axis=1)
        # t1 = time.time()
        # print(f'Halton constructin takes {t1-t0}')

            
        t1 = time.time()
        # print('Z takes {}'.format(t1-t0))
        n = X.shape[0]
        t0 = time.time()
        del X
        t1 = time.time()
        # print(f'delete X takes {t1-t0}')
        if method=='all':
            t0 = time.time()
            h = getfullComponent1(Xc, Z, y, center=center)
            t1 = time.time()
            SKAT_time = t1-t0
            # print(f'SKAT takes {t1-t0}')
            t0 = time.time()
            h2 = getfullComponent2(Xc, Z, y, center=center, RL_SKAT=True) 
            t1 = time.time()
            RL_SKAT_time = t1-t0
            # print(f'RL_SKAT takes {t1-t0}')
            return (([H[0] for H in h],SKAT_time,[H[1] for H in h]),([H[0] for H in h2],RL_SKAT_time,[H[1] for H in h]))
        elif method=='SKAT':
            t0 = time.time()
            h = getfullComponent1(Xc, Z, y, center=center)
            t1 = time.time()
            SKAT_time = t1-t0
            states = [H[1] for H in h]
            pvals = [H[0] for H in h]
            # print(f'SKAT takes {t1-t0}')
            return (pvals,SKAT_time,states)
        elif method=='Perm':
            t0 = time.time()
            h = getfullComponentPerm(Xc, Z, y, center=center, Test=Test, Perm=50)
            print(np.array(h).shape)
            # states = [H[1] for H in h]
            pvals1 = []
            pvals2 = []
            for phen in h:
                pvals1.append(phen[0][0])
                pvals2.append(phen[0][1])
            # print("CHI2")
            # print(pvals1)
            # print("FASTLMM")
            # print(pvals2)
            t1 = time.time()
            SKAT_time = t1-t0
            # print(f'SKAT Perm takes {t1-t0}')
            return (pvals1,pvals2,SKAT_time)

        elif method=='CCT' or method=='vary':
            t0 = time.time()
            h = getfullComponentPerm(Xc, Z, y, center=center,Perm=1,method='Scipy')
            states = [H[1] for H in h]
            pvals = [H[0] for H in h]
            t1 = time.time()
            SKAT_time = t1-t0
            # print(f'SKAT Perm takes {t1-t0}')
            return (pvals,SKAT_time,states)	

        elif method=='RL_SKAT':
            t0 = time.time()
            h2 = getfullComponent2(Xc, Z, y, center=center, RL_SKAT=True) 
            t1 = time.time()
            RL_SKAT_time = t1-t0
            # print(f'RL_SKAT takes {t1-t0}')
            return (h2,RL_SKAT_time)
        else:
            print(f'no method named {method}')
            return []

        

    elif how == 'fast_lin':
        print(f'Compute linear effect (SKAT)')
        center=params['center']
        Z = (X)/np.sqrt(X.shape[1]) 
        print(f'Z shape is {Z.shape}, Xc shape is {Xc.shape}')
        h = getfullComponent1(Xc, Z, y, center=center)
        return h

    elif how == 'quad':
        center=params['center']
        poly = PolynomialFeatures(2)
        Z = poly.fit_transform(X)
        Z = (Z)/np.sqrt(Z.shape[1]) 
        h = getfullComponent1(Xc, Z, y, center=center)
        return h

def impute_def(x):
    col_mean = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_mean, inds[1])
    return x

def impute(x):
    imp = SimpleImputer(missing_values=np.nan,strategy='mean')
    x = imp.fit_transform(x)
    return x
