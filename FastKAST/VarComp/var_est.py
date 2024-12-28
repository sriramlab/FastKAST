import time
import numpy as np
# from numba_stats import norm
from scipy.optimize import minimize
import numpy as np
from scipy.optimize import minimize
import time
from scipy.linalg import pinvh
from FastKAST.VarComp.se_est import *







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
    
