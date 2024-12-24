import numpy as np
import os, re
import time
from sys import path as syspath
from os import path as ospath
# from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import pairwise_kernels
# from sklearn.metrics.pairwise import additive_chi2_kernel
# from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.kernel_approximation import RBFSampler
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from fastmle_res_jax import getfullComponent as getfullComponent1
from fastmle_res_jax import getRLComponent as getfullComponent2
from fastmle_res_jax import getfullComponentPerm
from fastmle_res_jax import getmleComponent
from sklearn.impute import SimpleImputer
from utils import QMC_RFF
from sklearn.kernel_approximation import RBFSampler
import numpy as np
import traceback
from scipy.optimize import minimize
import sys
import gc
from scipy.linalg import svd
import time
from numpy.linalg import inv
import scipy
from tqdm import tqdm
from scipy.linalg import pinvh
import fastlmmclib.quadform as qf
from chi2comb import chi2comb_cdf, ChiSquared
from sklearn.linear_model import LogisticRegression
import scipy
from numpy.core.umath_tests import inner1d
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from utils import mix_chi_fit, mix_chi_quantile, fit_null







def estimateSigmasGeneral(y,
                          Xc,
                          X,
                          params=None,
                          how='rand_mom',
                          Random_state=1,
                          method='Perm',
                          Test='nonlinear'):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    if how == 'mle':
        gamma = params['gamma'] if params['gamma'] is not None else (
            1 / (X.shape[1] * X.var()))
        Kactual = pairwise_kernels(X,
                                   metric=params['kernel_metric'],
                                   gamma=gamma)
        t0 = time.time()
        center = params['center']
        h = getmleComponent(Xc, Kactual, y, center=center)
        t1 = time.time()
        Kernel_Time = t1 - t0
        states = [H[1] for H in h]
        pvals = [H[0] for H in h]
        print(f'mle pvals are: {pvals}')
        # print(f'get components takes {t1-t0}')
        return (pvals, Kernel_Time, states)

    elif how == 'fast_mle':
        gamma = params['gamma'] if params['gamma'] is not None else (
            1 / (X.shape[1] * X.var()))
        t0 = time.time()
        center = params['center']
        QMC = params['version']
        if QMC == 'Vanilla':
            rbfs = RBFSampler(gamma=gamma,
                              n_components=params['D'],
                              random_state=Random_state)
            Z = rbfs.fit_transform(X)
        else:
            Z = QMC_RFF(gamma=gamma,
                        d=X.shape[1],
                        n_components=params['D'],
                        seed=Random_state,
                        QMC=QMC).fit_transform(X)

        print(f'Test version is {Test}')

        t1 = time.time()
        n = X.shape[0]
        t0 = time.time()
        del X
        t1 = time.time()
        if method == 'SKAT':
            t0 = time.time()
            h = getfullComponent1(Xc, Z, y, center=center)
            t1 = time.time()
            SKAT_time = t1 - t0
            states = [H[1] for H in h]
            pvals = [H[0] for H in h]
            # print(f'SKAT takes {t1-t0}')
            return (pvals, SKAT_time, states)
        elif method == 'Perm':
            t0 = time.time()
            h = getfullComponentPerm(Xc,
                                     Z,
                                     y,
                                     center=center,
                                     Test=Test,
                                     Perm=10)
            p = h['pval']

            pvals = [H[0] for H in p]
            states = [H[1] for H in p]
            
            t1 = time.time()
            SKAT_time = t1 - t0
            # print(f'SKAT Perm takes {t1-t0}')
            return (pvals, SKAT_time, states)

        elif method == 'CCT' or method == 'vary':
            t0 = time.time()
            h = getfullComponentPerm(Xc,
                                     Z,
                                     y,
                                     center=center,
                                     Perm=1,
                                     method='Scipy')
            p = h['pval']
            pvals = [H[0] for H in p]
            states = [H[1] for H in p]
            t1 = time.time()
            SKAT_time = t1 - t0
            # print(f'SKAT Perm takes {t1-t0}')
            return (pvals, SKAT_time, states)

        elif method == 'RL_SKAT':
            t0 = time.time()
            h2 = getfullComponent2(Xc, Z, y, center=center, RL_SKAT=True)
            t1 = time.time()
            RL_SKAT_time = t1 - t0
            # print(f'RL_SKAT takes {t1-t0}')
            return (h2, RL_SKAT_time)
        else:
            print(f'no method named {method}')
            return []

    elif how == 'fast_lin':
        print(f'Compute linear effect (SKAT)')
        center = params['center']
        Z = (X) / np.sqrt(X.shape[1])
        print(f'Z shape is {Z.shape}, Xc shape is {Xc.shape}')
        h = getfullComponent1(Xc, Z, y, center=center)
        return h

    elif how == 'quad':
        center = params['center']
        poly = PolynomialFeatures(2)
        Z = poly.fit_transform(X)
        Z = (Z) / np.sqrt(Z.shape[1])
        h = getfullComponent1(Xc, Z, y, center=center)
        return h


