import numpy as np
import os, re
import time
from sys import path as syspath
from os import path as ospath
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import additive_chi2_kernel
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.kernel_approximation import RBFSampler
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from fastmle_res_jax import getfullComponent as getfullComponent1
from fastmle_res_jax import getRLComponent as getfullComponent2
from fastmle_res_jax import getfullComponentPerm
from fastmle_cali_jax import getmleComponent
from fastmle_res_jax import getmleComponent as getmleComponent2
from sklearn.impute import SimpleImputer
from RFF import RFF_fit_transform
def estimateSigmasGeneral(y, y_new, Xc, X, params=None, how='rand_mom',Random_state=1,method='SKAT'):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    X = preprocessing.scale(X)
    if how == 'rand_mom':
        gamma = params['gamma'] if params['gamma'] is not None else (1/(X.shape[1] * X.var()))
        rbfs = RBFSampler(gamma=gamma, n_components=params['D'])
        Z = rbfs.fit_transform(X)
        # comment this out to avoid normalizing stuff
        Z /= np.linalg.norm(Z, axis=1).reshape(X.shape[0], 1)
        h = fastEstimateSigmas(y, Z, hutReps=params['hutReps'])
        return h

   
    elif how == 'gemma':
        Kactual = pairwise_kernels(X, metric=params['kernel_metric'], gamma=params['gamma'])
        suffix = np.random.randint(0, 1000000)
        rFile = f'/tmp/relatedness_{suffix}'
        pFile = f'/tmp/pheno_{suffix}'
        resFile = f'/tmp/results_{suffix}'
        np.savetxt(rFile, Kactual, fmt='%.10f')
        np.savetxt(pFile, y.flatten(), fmt='%.10f')
        
        os.system(
            f'./gemma/gemma -p {pFile} -k {rFile} -n 1 -vc 1 -o results_{suffix} > {resFile}'
        )
        h = parse(resFile)
        os.system(f'rm {rFile}')
        os.system(f'rm {pFile}')
        return h
    
    elif how == 'rand_gemma':
        rbfs = RBFSampler(gamma=params['gamma'], n_components=params['D'])
        Z = rbfs.fit_transform(X)
        Kactual = Z.dot(Z.T)
        suffix = np.random.randint(0, 1000000)
        rFile = f'/tmp/relatedness_{suffix}'
        pFile = f'/tmp/pheno_{suffix}'
        resFile = f'/tmp/results_{suffix}'
        np.savetxt(rFile, Kactual, fmt='%.10f')
        np.savetxt(pFile, y.flatten(), fmt='%.10f')
        
        os.system(
            f'./gemma/gemma -p {pFile} -k {rFile} -n 1 -vc 1 -o results_{suffix} > {resFile}'
        )
        h = parse(resFile)
        os.system(f'rm {rFile}')
        os.system(f'rm {pFile}')
        return h

    elif how == 'mom':
        gamma = params['gamma'] if params['gamma'] is not None else (1/ (X.shape[1] * X.var()))
        Kactual = pairwise_kernels(X, metric=params['kernel_metric'], gamma=gamma)
        h = estimateSigmas(y, Kactual)
        return h

    elif how == 'rand_mle':
        gamma = params['gamma'] if params['gamma'] is not None else (1/ (X.shape[1] * X.var()))
        rbfs = RBFSampler(gamma=gamma, n_components=params['D'])
        Z = rbfs.fit_transform(X)
        Kapprox = Z.dot(Z.T)
        sigma = np.outer(y, y)
        N, _ = X.shape
        components = np.array([Kapprox, np.eye(N)])
        _, h, _ = reml(sigma, components)
        return h
    
    elif how == 'mle':
        gamma = params['gamma'] if params['gamma'] is not None else (1/ (X.shape[1] * X.var()))
        Kactual = pairwise_kernels(X, metric=params['kernel_metric'], gamma=gamma)
        t0 = time.time()
        center=params['center']
        h = getmleComponent(Xc, Kactual, y, center=center)
        t1 = time.time()
        Kernel_Time = t1-t0
        print(f'get components takes {t1-t0}')
        return (h,Kernel_Time)

    elif how == 'skat_mle':
        gamma = params['gamma'] if params['gamma'] is not None else (1/ (X.shape[1] * X.var()))
        Kactual = pairwise_kernels(X, metric=params['kernel_metric'], gamma=gamma)
        t0 = time.time()
        center=params['center']
        h = getmleComponent2(Xc, Kactual, y, center=center)
        t1 = time.time()
        Kernel_Time = t1-t0
        print(f'get components takes {t1-t0}')
        return (h,Kernel_Time)
 
    elif how == 'fast_mle':
        gamma = params['gamma'] if params['gamma'] is not None else (1/ (X.shape[1]*X.var()))
        print( gamma)
        t0 = time.time()
        center=params['center']
        try:
            if params['version'] == 1:
                rbfs = RBFSampler(gamma=gamma, n_components=params['D'],random_state=Random_state)
                Z = rbfs.fit_transform(X)
            elif params['version'] == 2:
                Z = RFF_fit_transform(X,params['D'],gamma,seed=Random_state)
        except:
            rbfs = RBFSampler(gamma=gamma, n_components=params['D'],random_state=Random_state)
            Z = np.float32(rbfs.fit_transform(X))
        t1 = time.time()
        print('Z takes {}'.format(t1-t0))
        n = X.shape[0]
        t0 = time.time()
        del X
        t1 = time.time()
        print(f'delete X takes {t1-t0}')
        if method=='all':
            t0 = time.time()
            h = getfullComponent1(Xc, Z, y, center=center)
            t1 = time.time()
            SKAT_time = t1-t0
            print(f'SKAT takes {t1-t0}')
            t0 = time.time()
            h2 = getfullComponent2(Xc, Z, y, center=center, RL_SKAT=True) 
            t1 = time.time()
            RL_SKAT_time = t1-t0
            print(f'RL_SKAT takes {t1-t0}')
            return ((h,SKAT_time),(h2,RL_SKAT_time))
        elif method=='SKAT':
            t0 = time.time()
            h = getfullComponent1(Xc, Z, y, center=center)
            t1 = time.time()
            SKAT_time = t1-t0
            print(f'SKAT takes {t1-t0}')
            return (h,SKAT_time)
        elif method=='Perm':
            t0 = time.time()
            h = getfullComponentPerm(Xc, Z, y, center=center)
            t1 = time.time()
            SKAT_time = t1-t0
            print(f'SKAT Perm takes {t1-t0}')
            return (h,SKAT_time)	    
        elif method=='RL_SKAT':
            t0 = time.time()
            h2 = getfullComponent2(Xc, Z, y, center=center, RL_SKAT=True) 
            t1 = time.time()
            RL_SKAT_time = t1-t0
            print(f'RL_SKAT takes {t1-t0}')
            return (h2,RL_SKAT_time)
        else:
            print(f'no method named {method}')
            return []

        

    elif how == 'fast_lin':
        center=params['center']
        Z = (X)/np.sqrt(X.shape[1]) 
        h = getfullComponent(Xc, Z, y, y_new, center=center)
        return h

    elif how == 'quad':
        center=params['center']
        poly = PolynomialFeatures(2)
        Z = poly.fit_transform(X)
        Z = (Z)/np.sqrt(Z.shape[1]) 
        h = getfullComponent1(Xc, Z, y, y_new, center=center)
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
