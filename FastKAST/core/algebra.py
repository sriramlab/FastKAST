import numpy as np
import time
import math
import pandas as pd
from tqdm import tqdm as tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import multiprocessing
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
from FastKAST.Test.stat_test import mix_chi_fit, mix_chi_quantile, fit_null





def _scipy_svd(X,compute_uv=False):
    return scipy.linalg.svd(X, full_matrices=False, compute_uv=compute_uv)

def _numpy_svd(X,compute_uv=False, full_matrices=False):
    return np.linalg.svd(X,full_matrices=full_matrices, compute_uv=compute_uv)

def _inverse_2(X):
    inverse = inv(X.T @ X)
    result = scipy.linalg.blas.sgemm(1., X.T, inverse.T, trans_a=True)
    return result


def _inverse(X):
    return pinvh(X.T @ X)  #change from pinv to inv sep 6
    # return pinvh(X.T@X)
    
def _projection(Z, X, P1):
    # Perform (I-X(X^TX)^-1 X^T)Z
    Z = np.array(Z, order='F')
    X = np.array(X, order='F')
    P1 = np.array(P1, order='F')
    t1 = scipy.linalg.blas.sgemm(1., X, Z, trans_a=True)
    t2 = scipy.linalg.blas.sgemm(1., X, P1)
    t3 = scipy.linalg.blas.sgemm(1., t2, t1)
    Z = Z - t3
    return Z


def _projection_2(Z, X, P1):
    Z = np.array(Z, order='F')
    X = np.array(X, order='F')
    P1 = np.array(P1, order='F')
    t1 = scipy.linalg.blas.sgemm(1., X, Z, trans_a=True)
    t3 = scipy.linalg.blas.sgemm(1., P1, t1)
    Z = Z - t3
    return Z


def _projection_mle(X, P1):
    X = np.array(X, order='F')
    P1 = np.array(P1, order='F')
    P1 = scipy.linalg.blas.sgemm(1., X, P1)
    P1 = scipy.linalg.blas.sgemm(1., P1, X, trans_b=True)
    return P1


def _PKP_comp(P, K):
    P = np.array(P, order='F')
    K = np.array(K, order='F')
    t1 = scipy.linalg.blas.sgemm(1., P, K)
    t2 = scipy.linalg.blas.sgemm(1., t1, P)
    return t2


