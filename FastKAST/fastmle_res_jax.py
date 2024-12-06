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

# def jax_svd(X):
#     return svd(X, full_matrices = False, compute_uv=False).block_until_ready()


def scipy_svd(X,compute_uv=False):
    return scipy.linalg.svd(X, full_matrices=False, compute_uv=compute_uv)

def numpy_svd(X,compute_uv=False, full_matrices=False):
    return np.linalg.svd(X,full_matrices=full_matrices, compute_uv=compute_uv)

def lik2(param, *args):
    if len(args) == 1:
        nargs = args
        (n, Sii, UTy, LLadd1) = nargs[0]
    else:
        (n, Sii, UTy, LLadd1) = args
    logdelta = param[0]
    gamma = param[1]
    # gamma = 0
    UTy = UTy.flatten()
    nulity = max(0, n - len(Sii))
    L1 = (sum(np.log(Sii * np.exp(gamma) + np.exp(logdelta))) +
          nulity * logdelta) / 2  # The first part of the log likelihood
    sUTy = np.square(UTy)
    if LLadd1 is None:
        # print('operation on L2')
        L2 = (n / 2.0) * np.log(
            (sum(sUTy / (Sii * np.exp(gamma) + np.exp(logdelta)))) / n)
    else:
        L2 = (n / 2.0) * np.log(
            (sum(sUTy / (Sii * np.exp(gamma) + np.exp(logdelta))) +
             (LLadd1 / (np.exp(logdelta)))) / n)
    return (L1 + L2)


def score_test(S):
    # N = Z.shape[0]
    k = len(S)
    dofs = np.ones(k)
    ncents = np.zeros(k)
    chi2s = [ChiSquared(S[i], ncents[i], dofs[i]) for i in range(k)]
    p, error, info = chi2comb_cdf(0, chi2s, 0, lim=10000000, atol=1e-14)
    # p = qf.qf(0, Phi, acc = 1e-7)[0]
    return (1 - p, error)


def score_test2(sq_sigma_e0, Q, S, decompose=True, center=False, multi=False, DEBUG=False):
    k = len(S)
    Phi = np.zeros(k)
    Phi[0:len(S)] = S
    Qe = Q / (sq_sigma_e0)
    dofs = np.ones(k)
    ncents = np.zeros(k)
    chi2s = [ChiSquared(Phi[i], ncents[i], dofs[i]) for i in range(k)]
    # t0 = time.time()
    if multi:
        ps=[]
        errors=[]
        infos=[]
        for K in tqdm(range(len(Qe)),desc="Processing score statistics"):
            p, error, info = chi2comb_cdf(Qe[K], chi2s, 0, lim=int(1e8), atol=1e-13)
            ps.append(p)
            errors.append(error)
            infos.append(info)
        if DEBUG:
            print(infos)
        ps = np.array(ps)
        return (1-ps, errors)
    else:
        p, error, info = chi2comb_cdf(Qe, chi2s, 0, lim=int(1e8), atol=1e-13)
        if DEBUG:
            print(info)
        # p = qf.qf(0, Phi, acc = 1e-7)[0]
        # t1 = time.time()
        return (1 - p, error)

def score_test_qf(sq_sigma_e0, Q, S, decompose=True, center=False,multi=False):
    Qe = (Q / (sq_sigma_e0))
    if multi:
        ps=[]
        for K in tqdm(range(len(Qe)),desc="Processing score statistics"):
            stats=qf.qf(Qe[K], S,sigma=1,lim=int(1e8),acc = 1e-15)
            p = stats[0]
            ps.append(p)
        ps = np.array(ps)
        return (ps,'NaN')
    else:
        stats=qf.qf(Qe, S,sigma=1,lim=int(1e8),acc = 1e-15)
        p = stats[0]
            
        return (p,'NaN')


def lik(logdelta, *args):
    if len(args) == 1:
        nargs = args
        (n, Sii, UTy, LLadd1) = nargs[0]
    else:
        (n, Sii, UTy, LLadd1) = args
    UTy = UTy.flatten()
    nulity = max(0, n - len(Sii))
    L1 = (sum(np.log(Sii + np.exp(logdelta))) +
          nulity * logdelta) / 2  # The first part of the log likelihood
    sUTy = np.square(UTy)
    if LLadd1 is None:
        L2 = (n / 2.0) * np.log((sum(sUTy / (Sii + np.exp(logdelta)))))
    else:
        L2 = (n / 2.0) * np.log((sum(sUTy / (Sii + np.exp(logdelta))) +
                                 (LLadd1 / (np.exp(logdelta)))))
    return (L1 + L2)


def lik_cov(logdelta, *args):
    if len(args) == 1:
        nargs = args
        (n, Sii, yt, LLadd1) = nargs[0]
    else:
        (n, Sii, yt, LLadd1) = args
    yt = yt.flatten()
    nulity = max(0, n - len(Sii)) ## N - K - K'
    L1 = (sum(np.log(Sii + np.exp(logdelta))) +
          nulity * logdelta) / 2  # The first part of the log likelihood
    syt = np.square(yt)

    L2 = (n / 2.0) * np.log((sum(syt / (Sii + np.exp(logdelta))) - sum(syt / np.exp(logdelta)) +
                                 (LLadd1 / (np.exp(logdelta)))))
    return (L1 + L2)


def standerr(U, y, Sii, UTy, g, e):
    L11 = np.sum(np.square(UTy.flatten()) * np.square(Sii) / (g * Sii + e)**3)
    L22 = np.sum(np.square(y - U @ UTy).flatten() / (e)**3) + np.sum(
        np.square(UTy).flatten() / ((e + Sii * g)**3))
    L12 = np.sum((np.square(UTy.flatten()) * Sii) / (g * Sii + e)**3)
    L = 0.5 * np.array([[L11, L12], [L12, L22]])
    cov = np.linalg.inv(L)
    gerr = np.sqrt(cov[0][0])
    eerr = np.sqrt(cov[1][1])
    return [gerr, eerr]

def standerr_dev(U, y, Sii, UTy, g, e):
    
    if cov:
        n = len(UTy)
        assert n>10
    else:
        n = len(y)
    print(g,e,n)
    # if isinstance(cov,int):
    #     n = len(y) - cov
    # else:
    #     n = len(y)
    nulity = max(0, n - len(Sii))
    L11 = -0.5*(np.sum(np.square(Sii) / (g * Sii + e)**2))+np.sum(np.square(UTy.flatten()) * np.square(Sii) / (g * Sii + e)**3) 
    L22 = -0.5*(np.sum(1. / (g * Sii + e)**2)+nulity*1./e**2) + np.sum(np.square(UTy.flatten()) / (g * Sii + e)**3) +  np.sum(np.square((y - U @ UTy).flatten()) / e**3)
    # print((np.square(UTy.flatten())).shape)
    # print(((g * Sii + e)**3).shape)
    # print(f'L22 is {L22}')
    L12 = -0.5*np.sum(Sii/np.square(g * Sii + e))+np.sum(np.square(UTy.flatten()) * Sii / (g * Sii + e)**3)
    # print(np.sum(np.square(UTy.flatten()) * Sii / (g * Sii + e)**3))
    # print(-0.5*np.sum(Sii/(g * Sii + e)**2))
    L = np.array([[L11, L12], [L12, L22]])
    # print(L)
    cov = np.linalg.inv(L)
    # print(cov)
    gerr = np.sqrt(cov[0][0])
    eerr = np.sqrt(cov[1][1])
    return [gerr, eerr]


def standerr_dev_cov(Sii, yt, LLadd1, n, g, e):
    '''
    This is the default standard deviation method to use
    '''
    nulity = max(0, n - len(Sii))
    # print(f'nulity is {nulity}')
    # print(f'Sii: {Sii}')
    # print(f'g: {g}')
    # print(f'e: {e}')
    L11 = -0.5*(np.sum(np.square(Sii) / np.square(g * Sii + e)))+np.sum(np.square(yt.flatten()) * np.square(Sii) / (g * Sii + e)**3) 
    L22 = -0.5*(np.sum(1. / np.square(g * Sii + e))+nulity*1./e**2) + np.sum(np.square(yt.flatten()) / (g * Sii + e)**3) - np.sum(np.square(yt.flatten())/e**3) +  LLadd1 / e**3
    L12 = -0.5*np.sum(Sii / np.square(g * Sii + e))+np.sum(np.square(yt.flatten()) * Sii / (g * Sii + e)**3)
    # print(np.sum(np.square(UTy.flatten()) * Sii / (g * Sii + e)**3))
    # print(-0.5*np.sum(Sii/(g * Sii + e)**2))
    L = np.array([[L11, L12], [L12, L22]])
    # print(f'L is:')
    # print(L)
    cov = np.linalg.inv(L)
    # print(f'Cov is:')
    # print(cov)
    gerr = np.sqrt(cov[0][0])
    eerr = np.sqrt(cov[1][1])
    return [gerr, eerr]


def dlik(logdelta, *args):
    n, Sii, UTy, LLadd1 = args
    UTy = UTy.flatten()
    delta = np.exp(logdelta)
    sUTy = np.square(UTy)
    if LLadd1 == None:
        LLadd1 = 0
    L1 = 0.5 * n * (np.sum(sUTy / np.square(Sii + delta)) +
                    LLadd1 / np.square(delta))
    L11 = np.sum(sUTy / (Sii + delta)) + LLadd1 / delta
    L2 = 0.5 * (np.sum(1 / (Sii + delta)) + (n - len(Sii)) * (1 / delta))
    der = np.zeros_like(delta)
    der[0] = -L1 / L11 + L2
    return der


def VarComponentEst(S, U, y, theta=False, dtype='quant',center=True,cov=False):
    # delta is the initial guess of delta value
    
    UTy = U.T @ y  # O(ND)

    n = y.shape[0]
    if n > len(S):
        LLadd1 = np.sum(np.square(y - U @ UTy))
    else:
        LLadd1 = None
    # optimizer = brent(lik, args=(n, S, UTy, LLadd1), brack = (-10, 10))
    t0 = time.time()
    optimizer = (minimize(lik, [0], args=(n, S, UTy, LLadd1), method = 'Nelder-Mead', options={'maxiter':400}))
    # optimizer = (minimize(lik, [0],
    #                       args=(n, S, UTy, LLadd1),
    #                       method='L-BFGS-B',
    #                       jac=dlik,
    #                       options={
    #                           'maxcor': 15,
    #                           'ftol': 1e-10,
    #                           'gtol': 1e-9,
    #                           'maxfun': 30000,
    #                           'maxiter': 30000,
    #                           'maxls': 30
    #                       }))
    logdelta = optimizer.x[0]
    t1 = time.time()
    # print(f'optimization takes {t1-t0}')
    # logdelta = optimizer
    # fun = -1*lik(logdelta, n, S, UTy, LLadd1)
    fun = -1 * optimizer.fun

    delta = np.exp(logdelta)
    h = 1 / (delta + 1)  # heritability
    if LLadd1 == None:
        sq_sigma_g = (sum(np.square(UTy.flatten()) / (S + delta))) / n
    else:
        sq_sigma_g = (sum(np.square(UTy.flatten()) /
                          (S + delta)) + LLadd1 / delta) / n

    sq_sigma_e = delta * sq_sigma_g
    time0 = time.time()
    gerr, eerr = standerr_dev(U, y, S, UTy, sq_sigma_g, sq_sigma_e, cov=cov)
    time1 = time.time()
    # print('error bound time is {}'.format(time1-time0))

    L1 = -lik(logdelta, n, S, UTy, LLadd1) - 0.5 * n * np.log(np.pi) - 0.5 * n
    yTy = (y.T @ y)[0]
    if dtype == 'quant':
        sq_sigma_e0 = yTy / n
    else:
        mu0 = np.sum(y) / n
        sq_sigma_e0 = mu0 * (1 - mu0)


#    sq_sigma_e0 = sq_sigma_e
    L0 = -0.5 * (n * np.log(np.pi) + n * np.log(sq_sigma_e0) +
                 yTy / sq_sigma_e0)
    return [
        h, sq_sigma_g, sq_sigma_e, gerr, eerr
    ]
   
   
   
def LRT(S, U, y, dtype='quant', Perm=0, Seed=None):
    '''
    ######### Boyang: Simple LRT without the inclusion of covariates #########
    
    :S: vector of shape (K')
    :yt: stands for transformed y. Shape (K') U.T@y
    :y1: stands for B1^Ty. Shape (K)
    :y: original trait. Shape (N)
    Perm: number of permutations (default is 0)
    '''
    # delta is the initial guess of delta value
    np.random.seed(Seed)
    num_iter=1 + Perm
    n = y.shape[0]
    y = y.copy()
    # print(f'U shape: {U.shape}; y shape: {y.shape}')
    yt = U.T@y
    LRT_stats = []
    for it in range(num_iter):

        if it > 0:
            y = np.random.permutation(y)
            yt = U.T@y
    
        LLadd1 = np.sum(np.square(y - U @ yt)) ## sum_{i=1}^{N-K} yt_i^2
    
        optimizer = (minimize(lik, [0], args=(n, S, yt, LLadd1), method = 'Nelder-Mead', options={'maxiter':5000}))
        logdelta = optimizer.x[0]
    
        fun = -1 * optimizer.fun
    
        delta = np.exp(logdelta)
        h = 1 / (delta + 1)  # heritability
        # print(h)
        
        sq_sigma_g = (sum(np.square(yt.flatten()) /
                              (S + delta)) + LLadd1 / delta) / n
        
        sq_sigma_e = delta * sq_sigma_g
    
        L1 = -lik(logdelta, n, S, yt, LLadd1) + 0.5*n*np.log(n) - 0.5 * n * np.log(2*np.pi) - 0.5*n
        # print(f'L1: {L1}')
        yTy = (y.T @ y)[0]
        if dtype == 'quant':
            sq_sigma_e0 = yTy / n
        else:
            mu0 = np.sum(y) / n
            sq_sigma_e0 = mu0 * (1 - mu0)
    
        L0 = -0.5 * (n * np.log(sq_sigma_e0) +
                     n)  - 0.5 * n * np.log(2*np.pi)

        LRT_stats.append(L1-L0[0])
        
    return np.array(LRT_stats)

       
def VarComponentEst_Cov_std(S, yt, y1, y, dtype='quant'):
    '''
    :S: vector of shape (K')
    :yt: stands for transformed y. Shape (K')
    :y1: stands for B1^Ty. Shape (K)
    :y: original trait. Shape (N)
    '''
    # delta is the initial guess of delta value
    
    
    n = y.shape[0]-y1.shape[0] ## N-K
    # print(f'n is {n}')
    
    LLadd1 = np.sum(np.square(y))-np.sum(np.square(y1)) ## sum_{i=1}^{N-K} yt_i^2
    
    ytilde_scale = np.sqrt(LLadd1/n)
    # print(f'y tilde std: {ytilde_scale}')
    S = S/ytilde_scale
    yt = yt/ytilde_scale
    
    # optimizer = brent(lik, args=(n, S, UTy, LLadd1), brack = (-10, 10))
    t0 = time.time()
    # optimizer = (minimize(lik_cov, [0], args=(n, S, yt, LLadd1), method = 'Nelder-Mead', options={'maxiter':400}))
    optimizer = (minimize(lik_cov, [0], args=(n, S, yt, n), method = 'Nelder-Mead', options={'maxiter':400}))
    logdelta = optimizer.x[0]
    t1 = time.time()
    # print(f'optimization takes {t1-t0}')
    # logdelta = optimizer
    # fun = -1*lik(logdelta, n, S, UTy, LLadd1)
    fun = -1 * optimizer.fun
    
    delta = np.exp(logdelta)
    # h = 1 / (delta + 1)  # heritability

    sq_sigma_g = (sum(np.square(yt.flatten()) / (S+delta)) - sum(np.square(yt.flatten()) / delta)  + n / delta) / n
    

    sq_sigma_e = delta * sq_sigma_g
    time0 = time.time()
    h = sq_sigma_g / (sq_sigma_e + sq_sigma_g)  # heritability
    # Sii, yt, LLadd1, n, g, e
    gerr, eerr = standerr_dev_cov(S, yt, n, n, sq_sigma_g, sq_sigma_e)
    time1 = time.time()
    # print('error bound time is {}'.format(time1-time0))

    yTy = (y.T @ y)[0]
    if dtype == 'quant':
        sq_sigma_e0 = yTy / n
    else:
        mu0 = np.sum(y) / n
        sq_sigma_e0 = mu0 * (1 - mu0)



    return [
        h, sq_sigma_g, sq_sigma_e, gerr, eerr
    ]
    
    
def VarComponentEst_Cov(S, yt, y1, y, dtype='quant'):
    '''
    :S: vector of shape (K')
    :yt: stands for transformed y. Shape (K')
    :y1: stands for B1^Ty. Shape (K)
    :y: original trait. Shape (N)
    '''
    # delta is the initial guess of delta value
    
    
    n = y.shape[0]-y1.shape[0] ## N-K
    
    LLadd1 = np.sum(np.square(y))-np.sum(np.square(y1)) ## sum_{i=1}^{N-K} yt_i^2
    
    
    # optimizer = brent(lik, args=(n, S, UTy, LLadd1), brack = (-10, 10))
    t0 = time.time()
    optimizer = (minimize(lik_cov, [0], args=(n, S, yt, LLadd1), method = 'Nelder-Mead', options={'maxiter':5000}))
    logdelta = optimizer.x[0]
    t1 = time.time()
    # print(f'optimization takes {t1-t0}')
    # logdelta = optimizer
    # fun = -1*lik(logdelta, n, S, UTy, LLadd1)
    fun = -1 * optimizer.fun

    delta = np.exp(logdelta)
    h = 1 / (delta + 1)  # heritability

    sq_sigma_g = (sum(np.square(yt.flatten()) / (S+delta)) - sum(np.square(yt.flatten()) / delta)  + LLadd1 / delta) / n
    
    # print(f'delta is: {delta}')
    sq_sigma_e = delta * sq_sigma_g
    time0 = time.time()
    # Sii, yt, LLadd1, n, g, e
    gerr, eerr = standerr_dev_cov(S, yt, LLadd1, n, sq_sigma_g, sq_sigma_e)
    time1 = time.time()
    # print('error bound time is {}'.format(time1-time0))

    L1 = -lik_cov(logdelta, n, S, yt, LLadd1) - 0.5 * n * np.log(np.pi) - 0.5 * n
    yTy = (y.T @ y)[0]
    if dtype == 'quant':
        sq_sigma_e0 = yTy / n
    else:
        mu0 = np.sum(y) / n
        sq_sigma_e0 = mu0 * (1 - mu0)


#    sq_sigma_e0 = sq_sigma_e
    L0 = -0.5 * (n * np.log(np.pi) + n * np.log(sq_sigma_e0) +
                 yTy / sq_sigma_e0)
    return [
        h, sq_sigma_g, sq_sigma_e, gerr, eerr
    ]


def projection(Z, X, P1):
    # Perform (I-X(X^TX)^-1 X^T)Z
    Z = np.array(Z, order='F')
    X = np.array(X, order='F')
    P1 = np.array(P1, order='F')
    t1 = scipy.linalg.blas.sgemm(1., X, Z, trans_a=True)
    t2 = scipy.linalg.blas.sgemm(1., X, P1)
    t3 = scipy.linalg.blas.sgemm(1., t2, t1)
    Z = Z - t3
    return Z


def projection_2(Z, X, P1):
    Z = np.array(Z, order='F')
    X = np.array(X, order='F')
    P1 = np.array(P1, order='F')
    t1 = scipy.linalg.blas.sgemm(1., X, Z, trans_a=True)
    t3 = scipy.linalg.blas.sgemm(1., P1, t1)
    Z = Z - t3
    return Z


def inverse_2(X):
    inverse = inv(X.T @ X)
    result = scipy.linalg.blas.sgemm(1., X.T, inverse.T, trans_a=True)
    return result


def inverse(X):
    return pinvh(X.T @ X)  #change from pinv to inv sep 6
    # return pinvh(X.T@X)


def getfullComponent(X, Z, y, dtype='quant', center=False, method='Scipy'):
    # X is the covariates that need to be regressed out, res is the residule after regressing out the linear effect
    # delta is the initial guess of delta value
    f1 = time.time()
    t0 = time.time()
    n = Z.shape[0]
    print(f'Z: {Z}')
    if X.size > 1:
        X = np.concatenate((np.ones((n, 1)), X), axis=1)
    else:
        X = np.ones(n, 1)
    y = y.reshape(-1, 1)
    k = X.shape[1]
    yperm = np.random.permutation(y)
    P1 = inverse(X)
    start = time.time()
    t1 = time.time()
    print(f'inverse P1 takes {t1-t0}')
    if center:
        print(f'SVD for PKP')
        t1 = time.time()
        Z = projection(Z, X, P1)
        t0 = time.time()
        print(f'Z operation takes {t1-t0}')
    
        S = numpy_svd(Z)
        # S = scipy.linalg.svd(Z, full_matrices=False, compute_uv=False)
        t1 = time.time()
        print(f'svd takes {t1-t0}')
        t0 = time.time()
        #        Q = np.sum(np.square(y.T@Z))
        Q = np.sum(np.square(y.T @ Z - y.T @ X @ P1 @ X.T @ Z))
        Q_perm = np.sum(np.square(yperm.T @ Z - yperm.T @ X @ P1 @ X.T @ Z))
        t1 = time.time()
    else:
        SVD = svd(Z.T @ Z)
        Q = np.sum(np.square(y.T @ Z))
    t0 = time.time()
    #     S = np.square(SVD[1])
    ts0 = time.time()
    S = np.square(S)
    S[S <= 1e-6] = 0
    S = S[np.nonzero(S)]
    S = S[~np.isnan(S)]
    print(f'S: {S}')
    print(f'Q: {Q}')
    ts1 = time.time()
    # k = int(np.sum(inner1d(P1,X)))
    t1 = time.time()
    if center:
        # sq_sigma_e0 = (res.T@res)[0]/(n-k)
        sq_sigma_e0 = (y.T @ y - y.T @ X @ P1 @ (X.T @ y))[0] / (n - k)
        sq_sigma_e0_perm = (yperm.T @ yperm -
                            yperm.T @ X @ P1 @ (X.T @ yperm))[0] / (n - k)
    else:
        sq_sigma_e0 = y.T @ y / n
    t0 = time.time()
    #   def score_test(sq_sigma_e0, Z, yres, S, decompose=True,center=False):
    if center:
        p_value1 = score_test2(sq_sigma_e0, Q, S, center=center)
        p_value1_perm = score_test2(sq_sigma_e0_perm, Q_perm, S, center=center)
    else:
        p_value1 = score_test2(sq_sigma_e0, Q, S, center=center)
    t1 = time.time()
    print(f'p value is {p_value1}, p_value1_perm is {p_value1_perm}')
    # print('e is {}'.format(sq_sigma_e0))
    return [p_value1, p_value1_perm]


def getfullComponentPerm(X,
                         Z,
                         y,
                         theta=False,
                         dtype='quant',
                         center=True,
                         method='Numpy',
                         Perm=10,
                         Test='nonlinear',
                         VarCompEst=False,
                         varCompStd=False):
    # X is the covariates that need to be regressed out, res is the residule after regressing out the linear effect
    # delta is the initial guess of delta value
    t0 = time.time()
    n = Z.shape[0]
    M = Z.shape[1]
    # print(f'Z here is {np.mean(Z,axis=0)}; {np.std(Z,axis=0)}')
    
    if center:
        if X is None or X.size==0:
            X = np.ones((n, 1))
        else:
            X = np.concatenate((np.ones((n, 1)), X), axis=1)
    y = y.reshape(-1, 1)
    if X is None or X.size==0:
        k = 0
        Q = np.sum(np.square(y.T @ Z))
        y1 = y.copy()
    else:
        k = X.shape[1]
    # yperm = np.random.permutation(y)
        P1 = inverse(X)
        # Z = left_projection(Z,X)
        # Z = projection_QR(Z,X,P1)
        Z = projection(Z, X, P1)
        Q = np.sum(np.square(y.T @ Z - y.T @ X @ P1 @ X.T @ Z))
        B1, _, _ = numpy_svd(X,compute_uv=True)
        y1 = B1.T@y
    # Z = Z - X@P1@(X.T@Z)
    t1 = time.time()
    # print(f'Z operation takes {t1-t0}')
    t0 = time.time()
    if VarCompEst:
        U,S,_ = numpy_svd(Z,compute_uv=True)
    else:
        S = numpy_svd(Z)
    # S = scipy.linalg.svd(Z, full_matrices=False, compute_uv=False)

    t1 = time.time()
    # print(f'svd takes {t1-t0}')
    # t0 = time.time()

    # Q_perm = np.sum(np.square(yperm.T@Z - yperm.T@X@P1@X.T@Z))
    # t1 = time.time()
    S = np.square(S)
    S[S <= 1e-6] = 0
    filtered=np.nonzero(S)[0]
    S = S[filtered]

    results = {}
    if VarCompEst:
        # print(U.shape)
        U = U[:,filtered]
        # print(U.shape,S.shape)
        if X is None:
            var_est=VarComponentEst(S,U,y)
        else:
            yt = U.T@y
            if varCompStd:
                # standardized hertability for sigma_quad^2
                var_est=VarComponentEst_Cov_std(S,yt,y1,y) # def VarComponentEst_Cov(S, yt, y1, y, dtype='quant'):
            else:
                var_est=VarComponentEst_Cov(S,yt,y1,y) # def VarComponentEst_Cov(S, yt, y1, y, dtype='quant'):
        sigma2_gxg=var_est[1]
        sigma2_e=var_est[2]
        trace=np.sum(S) # compute the trace of phi phi.T
        sumK = np.sum(np.sum(Z,axis=0)**2) # compute the sum(Phi Phi.T)
        print(f'trace is {trace}; sum K is {sumK}')

        # cC=trace*1.0/(n*M) - sumK*1.0/(n**2*M)
        # print(f'Constant factor is {cC}')
        # h2_gxg=cC*sigma2_gxg/(cC*sigma2_gxg+((n-1)*1.0/n)*sigma2_e)
        # print(f'Before correction: {sigma2_gxg}; after correction: {h2_gxg}')
        results['varcomp']=var_est
        print(f'Var est is: \n {var_est}')
    t0 = time.time()
    #     S = np.square(SVD[1])
    ts0 = time.time()
    
    # print(f'S raw is {S}')
    
    ts1 = time.time()
    # k = int(np.sum(inner1d(P1,X)))
    t1 = time.time()
    if center:
        # print('calculate centered y')
        # sq_sigma_e0 = (res.T@res)[0]/(n-k)
        sq_sigma_e0 = (y.T @ y - y.T @ X @ P1 @ (X.T @ y))[0] / (n - k)
        # sq_sigma_e0_perm = (yperm.T@yperm - yperm.T@X@P1@(X.T@yperm))[0]/(n-k)
    else:
        sq_sigma_e0 = y.T @ y / n
    # t0 = time.time()
    # print(f'Y is {y}, {np.sum(y)}')
    
    p_value1 = score_test2(sq_sigma_e0, Q, S, center=center)
    # print(f'pval is {p_value1}')
#     p_values2 = score_test_qf(sq_sigma_e0, Q, S, center=center)
#     print(f'chi2comb pval: {p_value1} \n FastLMM pval: {p_values2}')
    # print(f'Q is {Q}; sq_sigma_e0 is {sq_sigma_e0}; pval is {p_value1}')
    if Perm:
        p_list = [p_value1]
        for state in range(Perm):
            shuff_idx = np.random.RandomState(seed=state).permutation(n)
            yperm = (y - (X @ (P1 @ (X.T @ y))))[shuff_idx]
            Qperm = np.sum(np.square(yperm.T @ Z))
            sq_sigma_e0_perm = (yperm.T @ yperm)[0] / (n - k)
            p_value1_perm = score_test2(sq_sigma_e0_perm,
                                        Qperm,
                                        S,
                                        center=center)
            p_list.append(p_value1_perm)

        results['pval']=p_list
        print(f'results is {results}')
        return results
    results['pval']=p_value1
    return results



def getfullComponentMulti(X,
                         Z,
                         y,
                         theta=False,
                         dtype='quant',
                         center=True,
                         method='Numpy',
                         Perm=10,
                         Test='nonlinear',
                         VarCompEst=False,
                         varCompStd=False):
    '''
    This function provide a multi-trait version for QuadKAST computation.
    Detailed processing scheme
        1. Trait missing value will be masked
        2. Effective sample size N varies trait by trait
    Input:
        y: N x K (K: number of traits to test)
        X: covariates
        Z: (transformed) genotype matrix
    '''
    # X is the covariates that need to be regressed out, res is the residule after regressing out the linear effect
    # delta is the initial guess of delta value
    t0 = time.time()
    n = Z.shape[0]
    M = Z.shape[1]
    # print(f'Z here is {np.mean(Z,axis=0)}; {np.std(Z,axis=0)}')
    nan_num = np.sum(np.isnan(y),axis=0)
    print(f'nan_num is {nan_num}')
    y = np.nan_to_num(y)
    if center:
        if X is None or X.size==0:
            X = np.ones((n, 1))
        else:
            X = np.concatenate((np.ones((n, 1)), X), axis=1)
    
    if X is None or X.size==0:
        k = 0
        Q = np.sum(np.square(y.T @ Z),axis=1) ## K vector 
        y1 = y.copy()
    else:
        k = X.shape[1]
    # yperm = np.random.permutation(y)
        P1 = inverse(X)
        # Z = left_projection(Z,X)
        # Z = projection_QR(Z,X,P1)
        Z = projection(Z, X, P1)
        Q = np.sum(np.square(y.T @ Z - y.T @ X @ P1 @ X.T @ Z),axis=1) ## K vector
        B1, _, _ = numpy_svd(X,compute_uv=True) ## N_eff x N
        y1 = B1.T@y ## N_eff x K
    # Z = Z - X@P1@(X.T@Z)
    t1 = time.time()
    # print(f'Z operation takes {t1-t0}')
    if VarCompEst:
        U,S,_ = numpy_svd(Z,compute_uv=True)
    else:
        S = numpy_svd(Z)
    # S = scipy.linalg.svd(Z, full_matrices=False, compute_uv=False)

    t1 = time.time()
    # print(f'svd takes {t1-t0}')
    # t0 = time.time()

    # Q_perm = np.sum(np.square(yperm.T@Z - yperm.T@X@P1@X.T@Z))
    # t1 = time.time()
    S = np.square(S)
    S[S <= 1e-6] = 0
    filtered=np.nonzero(S)[0]
    S = S[filtered]

    results = {}
    if VarCompEst:
        print(f'Var comp for multi-trait version hasnt implemented yet')
        # print(U.shape)
        U = U[:,filtered]
        # print(U.shape,S.shape)
        if X is None:
            var_est=VarComponentEst(S,U,y)
        else:
            yt = U.T@y
            if varCompStd:
                # standardized hertability for sigma_quad^2
                var_est=VarComponentEst_Cov_std(S,yt,y1,y) # def VarComponentEst_Cov(S, yt, y1, y, dtype='quant'): Don't use this version unless you know what you are doing
            else:
                var_est=VarComponentEst_Cov(S,yt,y1,y) # def VarComponentEst_Cov(S, yt, y1, y, dtype='quant'):
        sigma2_gxg=var_est[1]
        sigma2_e=var_est[2]
        trace=np.sum(S) # compute the trace of phi phi.T
        sumK = np.sum(np.sum(Z,axis=0)**2) # compute the sum(Phi Phi.T)
        # print(f'trace is {trace}; sum K is {sumK}')
        results['varcomp']=var_est
        print(f'Var est is: \n {var_est}')
    t0 = time.time()
    #     S = np.square(SVD[1])
    ts0 = time.time()
    
    # print(f'S raw is {S}')
    
    ts1 = time.time()
    # k = int(np.sum(inner1d(P1,X)))
    t1 = time.time()
    
    if center:
        # print('calculate centered y')
        # sq_sigma_e0 = (y.T @ y - y.T @ X @ P1 @ (X.T @ y))[0] / (n - k)
        yTXPX = y.T @ X @ P1 @ X.T
        sq_sigma_e0_num=np.sum(y*y,axis=0) - np.sum(yTXPX*(y.T),axis=1)
        sq_sigma_e0_den=(n-k-nan_num) 
        sq_sigma_e0 = sq_sigma_e0_num / sq_sigma_e0_den ## K vector
        # print(f'sq_sigma_e0: {sq_sigma_e0}')
        # sq_sigma_e0_perm = (yperm.T@yperm - yperm.T@X@P1@(X.T@yperm))[0]/(n-k)
    else:
        sq_sigma_e0 = np.sum(y*y,axis=0) / (n-nan_num) ## K vector
    # t0 = time.time()
    # print(f'Y is {y}, {np.sum(y)}')
    p_value1=score_test_qf(sq_sigma_e0, Q, S, center=center,multi=True)
    # p_value1 = score_test2(sq_sigma_e0, Q, S, center=center,multi=True)
    # print(f'pval is {p_value1}')
    if Perm:
        print(f'Perm not implement yet')
        pass
        # p_list = [p_value1]
        # for state in range(Perm):
        #     shuff_idx = np.random.RandomState(seed=state).permutation(n)
        #     yperm = (y - (X @ (P1 @ (X.T @ y))))[shuff_idx]
        #     Qperm = np.sum(np.square(yperm.T @ Z))
        #     sq_sigma_e0_perm = (yperm.T @ yperm)[0] / (n - k)
        #     p_value1_perm = score_test2(sq_sigma_e0_perm,
        #                                 Qperm,
        #                                 S,
        #                                 center=center)
        #     p_list.append(p_value1_perm)

        # results['pval']=p_list
        # return results
    results['pvals']=p_value1 ## notice the key name change here
    return results


##########
# Update the binary trait process

# def getfullComponentPerm_binary(X, Z, y, center=False,method='Scipy',Perm=10):
#     # X is the covariates that need to be regressed out, res is the residule after regressing out the linear effect
#     # delta is the initial guess of delta value
#     print(f'use {method}')

#     t0 = time.time()
#     n = Z.shape[0]
#     X = np.concatenate((np.ones((n,1)),X),axis=1)
#     y = y.reshape(-1,1)
#     clf = LogisticRegression(random_state=0,fit_intercept=False).fit(X, y)
#     est_mu = clf.predict_proba(X)
#     k = X.shape[1]
#     # yperm = np.random.permutation(y)
#     P1= inverse(X)
#     t1 = time.time()
#     # print(f'inverse P1 takes {t1-t0}')
#     if center:
#         # S = svd(Z.T@Z-(Z.T@P1)@(X.T@Z),compute_uv=False)
#         t0 = time.time()
#         Z = projection(Z,X,P1)
#         # Z = Z - X@P1@(X.T@Z)
#         t1 = time.time()
#         # print(f'Z operation takes {t1-t0}')
#         if method == 'Jax':
#             S = jax_svd(Z)
#         elif method == 'Julia':
#             if Julia_FLAG:
#                 S = FameSVD.fsvd(Z).S
#             else:
#                 S = scipy.linalg.svd(Z,full_matrices = False, compute_uv=False)
#         elif method == 'Scipy':
#             S = scipy_svd(Z)

#         Q = np.sum(np.square(y.T@Z - y.T@X@P1@X.T@Z))

#         t1 = time.time()
#         print(f'svd takes {t1-t0}')
#         t0 = time.time()

#         # Q_perm = np.sum(np.square(yperm.T@Z - yperm.T@X@P1@X.T@Z))
#         t1 = time.time()
#     else:
#         SVD = svd(Z.T@Z)
#         Q = np.sum(np.square(y.T@Z))
#     t0 = time.time()
# #     S = np.square(SVD[1])
#     ts0 = time.time()
#     S = np.square(S)
#     S[S <= 1e-6] = 0
#     S = S[np.nonzero(S)]
#     # S = S[~np.isnan(S)]
#     ts1 = time.time()
#     # k = int(np.sum(inner1d(P1,X)))
#     t1 = time.time()
#     if center:
#         # print('calculate centered y')
#         # sq_sigma_e0 = (res.T@res)[0]/(n-k)
#         sq_sigma_e0 = (y.T@y - y.T@X@P1@(X.T@y))[0]/(n-k)
#         # sq_sigma_e0_perm = (yperm.T@yperm - yperm.T@X@P1@(X.T@yperm))[0]/(n-k)
#     else:
#         sq_sigma_e0 = y.T@y/n
#     # t0 = time.time()
#     p_value1 = score_test2(sq_sigma_e0, Q, S, center=center)
#     if Perm:
#         p_list = [p_value1]
#         for state in range(Perm):
#             shuff_idx = np.random.RandomState(seed=state).permutation(n)
#             yperm = (y-(X@(P1@(X.T@y))))[shuff_idx]
#             Qperm = np.sum(np.square(yperm.T@Z))
#             sq_sigma_e0_perm = (yperm.T@yperm)[0]/(n-k)
#             p_value1_perm = score_test2(sq_sigma_e0_perm, Qperm, S, center=center)
#             p_list.append(p_value1_perm)
#         # t1 = time.time()
#         # print(f'p value test takes {t1-t0}')
#         return p_list

#     return p_value1

# Started on Jun 13th
################


def getRLComponent(X,
                   Z,
                   y,
                   theta=False,
                   dtype='quant',
                   center=False,
                   RL_SKAT=True,
                   method='Julia'):
    # X is the covariates that need to be regressed out, res is the residule after regressing out the linear effect
    # delta is the initial guess of delta value

    t0 = time.time()
    n = Z.shape[0]
    X = np.concatenate((np.ones((n, 1)), X), axis=1)
    k = X.shape[1]
    yperm = np.random.permutation(y)
    P1 = inverse(X)
    t1 = time.time()
    p = X.shape[1]
    if center:
        t0 = time.time()
        Z = projection(Z, X, P1)
        t1 = time.time()
        # print(f'Z operation takes {t1-t0}')
        S = scipy.linalg.svd(X, full_matrices=False, compute_uv=False)
        S = np.square(S)
        S[S <= 1e-6] = 0
        S = S[np.nonzero(S)]
        S = S[~np.isnan(S)]
        sq_sigma_e0 = (y.T @ y - y.T @ X @ P1 @ (X.T @ y)) / (n - p)
        sq_sigma_e0_perm = (yperm.T @ yperm -
                            yperm.T @ X @ P1 @ (X.T @ yperm)) / (n - p)
        t1 = time.time()
        Q = (y.T @ Z) @ (Z.T @ y)
        Q_perm = (yperm.T @ Z) @ (Z.T @ yperm)
        # print("svd takes {}".format(t1-t0))
        Qe = Q / (sq_sigma_e0)
        Qe_perm = Q_perm / (sq_sigma_e0_perm)
        if RL_SKAT:
            # under the assumption that PZ and X has no overlapping
            # C = np.concatenate((Z,X),axis=1)
            t0 = time.time()
            t1 = time.time()
            # print(f'conversion takes {t1-t0}')
            phi = S
            t2 = time.time()
            rankPZ = len(phi)
            rankX = X.shape[1]
            # rankX = np.linalg.matrix_rank(X)
            q = n - rankPZ - rankX
            t1 = time.time()
            k = len(phi)
            S = np.zeros(k + q)
            S[0:k] = phi
            S_perm = S - Qe_perm / (n - p)
            S -= Qe / (n - p)
        else:
            S = S[1]
            S = np.square(SVD[1])
            S[np.abs(S) < 1e-6] = 0
            S = S[np.nonzero(S)]
    else:
        Q = (y.T @ Z) @ (Z.T @ y)
        Qe = Q / (sq_sigma_e0)
        sq_sigma_e0 = y.T @ y / n
        # print('SVD for K')
        SVD = svd(Z, full_matrices=False)
    y = y.reshape(-1, 1)
    # print(f'rank of XinvXT is {p}')
    t0 = time.time()
    if center:
        p_value1 = score_test(S)
        p_value1_perm = score_test(S_perm)
    else:
        p_value1 = score_test(S)
    t1 = time.time()
    # print(f'total p value time is {t1-t0}')
    return [p_value1, p_value1_perm]


def projection_mle(X, P1):
    X = np.array(X, order='F')
    P1 = np.array(P1, order='F')
    P1 = scipy.linalg.blas.sgemm(1., X, P1)
    P1 = scipy.linalg.blas.sgemm(1., P1, X, trans_b=True)
    return P1


def PKP_comp(P, K):
    P = np.array(P, order='F')
    K = np.array(K, order='F')
    t1 = scipy.linalg.blas.sgemm(1., P, K)
    t2 = scipy.linalg.blas.sgemm(1., t1, P)
    return t2


def getmleComponent(X, K, y, center=False):
    # delta is the initial guess of delta value
    t0 = time.time()
    n = K.shape[0]
    y = y.reshape(-1, 1)
    yperm = np.random.permutation(y)
    X = np.concatenate((np.ones((n, 1)), X), axis=1)
    # P1= X@np.linalg.inv(X.T@X)@X.T
    # P = np.eye(n)-P1
    # PKP = P@K@P
    P1 = inverse(X)
    P1 = projection_mle(X, P1)
    P = np.eye(n) - P1
    PKP = PKP_comp(P, K)
    Q = y.T @ PKP @ y
    Q_perm = yperm.T @ PKP @ yperm
    try:
        # S = jax_svd(PKP)
        S = svd(PKP, full_matrices=False, compute_uv=False)
    except:
        print(f'X shape is {X.shape}')
        print(f'P1 shape is {P1.shape}')
        print(f'PKP contains NA: {np.isnan(np.sum(PKP))}')
        return []
    t1 = time.time()
    print("svd takes {}".format(t1 - t0))
    S = S[S >= 1e-6]
    S = S[np.nonzero(S)]
    t0 = time.time()
    k = X.shape[1]
    t1 = time.time()

    if center:
        sq_sigma_e0 = (y.T @ y - (y.T @ P1 @ y))[0] / (n - k)
        sq_sigma_e0_perm = (yperm.T @ yperm -
                            (yperm.T @ P1 @ yperm))[0] / (n - k)
    else:
        sq_sigma_e0 = y.T @ y / n
    t0 = time.time()
    if center:
        p_value1 = score_test2(sq_sigma_e0,
                               Q,
                               S,
                               center=center,
                               decompose=False)
        p_value_perm = score_test2(sq_sigma_e0_perm,
                                   Q_perm,
                                   S,
                                   center=center,
                                   decompose=False)
    else:
        p_value1 = score_test2(sq_sigma_e0,
                               Q,
                               S,
                               center=center,
                               decompose=False)
    t1 = time.time()
    print(f'p value is {p_value1}')
    return [p_value1, p_value_perm]


def Bayesian_Posterior(X,Z,y,g,e,center=True,full_cov=False):
    '''
    Feature level posterior estimation
    Assume Z = Phi(G)/sqrt(D)
    '''
    
    t0 = time.time()
    n = Z.shape[0]
    D = Z.shape[1]
    Z = Z*np.sqrt(D)
    
    if center:
        if X is None or X.size==0:
            X = np.ones((n, 1))
        else:
            X = np.concatenate((np.ones((n, 1)), X), axis=1)
            
            
    y = y.reshape(-1, 1)

    # yperm = np.random.permutation(y)
    P1 = inverse(X)

    Z = projection(Z, X, P1) # Z = Z - X@P1@(X.T@Z)
    
    y_proj = projection(y,X,P1)
    
    inverse_factor = pinvh(Z.T@Z + np.identity(D)*(D*e/g)) # (phi(G)^T phi(G)) + reg*I_D)^{-1}
    
    mu = (inverse_factor)@(Z.T@y_proj)
    mu = mu.flatten()
    cov = e*inverse_factor
    
    
    if full_cov:
        return mu, cov
    else:
        return mu, np.sqrt(np.diag(cov))
    


if __name__ == "__main__":
    import statsmodels.api as sm
    from regressors import stats
    import scipy.stats
    results = []
    dtype = 'quant'
    np.random.seed(1)
    from sklearn import preprocessing
    from sklearn.kernel_approximation import PolynomialCountSketch
    print(f'Simulating linear effect with h2 = 0.5')
    for sigma1sq, sigma2sq in [(0.1, 0.9)]:
        N = 5000
        M = 20
        D = M * 50
        gamma = 0.1
        X = np.random.binomial(2, np.random.uniform(0.1, 0.5, M), (N, M))
        

        mapping = PolynomialFeatures((2, 2),interaction_only=False,include_bias=False)
        for i in range(3,6):
            sigmalinsq=0.4
            Z = mapping.fit_transform(X)
            Z = preprocessing.scale(Z)
            print(f'Z shape is {Z.shape}')
            eps = np.random.randn(N) * np.sqrt(sigma2sq)
            beta = np.random.randn(Z.shape[1]) * np.sqrt(sigma1sq)*1.0/np.sqrt(Z.shape[1])
            alpha =  np.random.randn(X.shape[1]) * np.sqrt(sigmalinsq)*1.0/np.sqrt(X.shape[1])
            y = Z.dot(beta) + eps
            print(f'y var is {np.var(y)}')
            # y += X.dot(alpha)
            print(f'y var is {np.var(y)}')
            # plist = getfullComponent(X,
            #                          Z,
            #                          y,
            #                          dtype=dtype,
            #                          center=True,
            #                          method="Julia")
            # print(f'FastKAST p value is {plist[0][0]}')
            # results = getfullComponentPerm(None,Z*1.0/np.sqrt(Z.shape[1]),y.reshape(1,-1),VarCompEst=True)
            results = getfullComponentPerm(X,Z*1.0/np.sqrt(Z.shape[1]),y.reshape(1,-1),VarCompEst=True,varCompStd=False)
            g, e = results['varcomp'][1], results['varcomp'][2]
            print(f'g_est: {g}; e_est: {e}')
            mu, cov = Bayesian_Posterior(X,Z*1.0/np.sqrt(Z.shape[1]),y,g,e)
            # print(f'True betas: {beta}')
            # print(f'beta_est: {mu}; std_est: {cov}')
            p_value = scipy.stats.norm.sf(abs(mu/cov))*2
            # print(f'Ridge p_values: {p_value}')
            
            
            reg_1 = LinearRegression()
            y_res = y - reg_1.fit(X, y).predict(X)
            reg = Ridge(alpha=Z.shape[1]*e/g)
            reg.fit(Z,y_res)
           #  print(f'Ridge coeff: {reg.coef_}')
            
            pvals_ridge=stats.coef_pval(reg, Z, y_res)
            # print(f'Ridge ground pval: {pvals_ridge[1:]}')
            
            model = sm.OLS(y_res, Z)
            model = model.fit()
            OLS_pvalues=model.pvalues
           #  print(f'OLS pvals: {OLS_pvalues}')
            # print(results)
            # results.append((plist, sigma1sq / (sigma1sq + sigma2sq), N, M, D))

    # dump(results, f'./test.pkl')
