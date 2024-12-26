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
