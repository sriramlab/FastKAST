import time

import numpy as np
import scipy
from FastKAST.core.algebra import *
from FastKAST.core.algebra import _inverse, _numpy_svd, _projection
from FastKAST.stat_test.stat_test import score_test2
from FastKAST.core.optim import *
from FastKAST.VarComp.se_est import *
from FastKAST.VarComp.var_est import *
from scipy.linalg import svd
from scipy.optimize import minimize


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
        P1 = _inverse(X)
        # Z = left__projection(Z,X)
        # Z = _projection_QR(Z,X,P1)
        Z = __projection(Z, X, P1)
        Q = np.sum(np.square(y.T @ Z - y.T @ X @ P1 @ X.T @ Z),axis=1) ## K vector
        B1, _, _ = _numpy_svd(X,compute_uv=True) ## N_eff x N
        y1 = B1.T@y ## N_eff x K
    # Z = Z - X@P1@(X.T@Z)
    t1 = time.time()
    # print(f'Z operation takes {t1-t0}')
    if VarCompEst:
        U,S,_ = _numpy_svd(Z,compute_uv=True)
    else:
        S = _numpy_svd(Z)
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
        P1 = _inverse(X)
        # Z = left__projection(Z,X)
        # Z = _projection_QR(Z,X,P1)
        Z = _projection(Z, X, P1)
        Q = np.sum(np.square(y.T @ Z - y.T @ X @ P1 @ X.T @ Z))
        B1, _, _ = _numpy_svd(X,compute_uv=True)
        y1 = B1.T@y
    # Z = Z - X@P1@(X.T@Z)
    t1 = time.time()
    # print(f'Z operation takes {t1-t0}')
    t0 = time.time()
    if VarCompEst:
        U,S,_ = _numpy_svd(Z,compute_uv=True)
    else:
        S = _numpy_svd(Z)
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
    P1 = _inverse(X)
    start = time.time()
    t1 = time.time()
    print(f'_inverse P1 takes {t1-t0}')
    if center:
        print(f'SVD for PKP')
        t1 = time.time()
        Z = _projection(Z, X, P1)
        t0 = time.time()
        print(f'Z operation takes {t1-t0}')
    
        S = _numpy_svd(Z)
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
    P1 = _inverse(X)
    t1 = time.time()
    p = X.shape[1]
    if center:
        t0 = time.time()
        Z = _projection(Z, X, P1)
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
    P1 = _inverse(X)
    P1 = _projection_mle(X, P1)
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
