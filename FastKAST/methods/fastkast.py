import time

import numpy as np
import scipy
from scipy.linalg import svd
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
from FastKAST.core.algebra import *
from FastKAST.core.algebra import _inverse, _numpy_svd, _projection
from FastKAST.core.optim import _lik
from FastKAST.stat_test.stat_test import score_test2, score_test_qf
from FastKAST.core.optim import *
from FastKAST.VarComp.se_est import *
from FastKAST.VarComp.var_est import *


class FastKASTComponent:
    """
    Class for performing FastKAST component analysis.

    This class encapsulates the logic for computing the FastKAST component, 
    including variance component estimation and p-value calculation.
    """

    def __init__(self, X, Z, y, theta=False, dtype='quant', center=True,
                 method='Numpy', Perm=10, MapFunc='linear',
                 VarCompEst=False, varCompStd=False, D=10, gamma=0.1, Random_state=0, mapping=None):
        """
        Initialize the FastKASTComponent object.

        Args:
            X (np.ndarray): Covariates to be regressed out. -- If intend to test 
            Z (np.ndarray): Genotype (transformed) matrix.
            y (np.ndarray): Phenotype vector.
            theta (bool, optional): Whether to include theta in the model. Defaults to False.
            dtype (str, optional): Data type. Defaults to 'quant'.
            center (bool, optional): Whether to center the data. Defaults to True.
            method (str, optional): Method for computation. Defaults to 'Numpy'.
            Perm (int, optional): Number of permutations for p-value calculation. Defaults to 10.
            MapFunc (str, optional): Function used to map the feature. Defaults to 'nonlinear'.
            VarCompEst (bool, optional): Whether to estimate variance components. Defaults to False.
            varCompStd (bool, optional): Whether to standardize variance components. Defaults to False.
        """
        self.X = X
        self.Z = Z
        self.y = y
        self.theta = theta
        self.dtype = dtype
        self.center = center
        self.method = method
        self.Perm = Perm
        self.MapFunc = MapFunc
        self.VarCompEst = VarCompEst
        self.varCompStd = varCompStd
        self.D = D
        self.gamma = gamma
        self.mapping = mapping
        self.Random_state = Random_state
        self.n = Z.shape[0]
        self.M = Z.shape[1]

    def _preprocess_data(self):
        """
        Preprocess the data by centering and projecting.
        """
        if self.center:
            if self.X is None or self.X.size == 0:
                self.X = np.ones((self.n, 1))
            else:
                self.X = np.concatenate((np.ones((self.n, 1)), self.X), axis=1)
        self.y = self.y.reshape(-1, 1)

        if self.X is None or self.X.size == 0:
            self.k = 0
            self.Q = np.sum(np.square(self.y.T @ self.Z))
            self.y1 = self.y.copy()
        else:
            self.k = self.X.shape[1]
            P1 = _inverse(self.X)
            self.P1 = P1
            self.Z = _projection(self.Z, self.X, P1)
            self.Q = np.sum(np.square(self.y.T @ self.Z -
                            self.y.T @ self.X @ P1 @ self.X.T @ self.Z))
            B1, _, _ = _numpy_svd(self.X, compute_uv=True)
            self.y1 = B1.T @ self.y

    def _feature_map(self):
        """
        Perform feature transformation depending on the request
        """
        if self.mapping is not None:
            # Use customized mapping function
            self.Z = self.mapping(self.Z)

        if self.MapFunc == 'linear':
            # Use linear mapping function
            self.Z = (self.Z) / np.sqrt(self.Z.shape[1])

        elif self.MapFunc == 'rbf':
            # Use RBF approximation
            mapping = RBFSampler(gamma=self.gamma,
                                 n_components=self.D,
                                 random_state=self.Random_state)
            self.mapping = mapping
            Z = self.mapping.fit_transform(self.Z)
            self.Z = (Z) / np.sqrt(Z.shape[1])

        elif self.MapFunc == 'quadOnly':
            # Use quadratic only feature map
            mapping = PolynomialFeatures(
                (2, 2), interaction_only=True, include_bias=False)
            self.mapping = mapping
            Z = mapping.fit_transform(self.Z)
            self.Z = Z

        else:
            raise NotImplementedError(f"{self.MapFunc} is not implemented")

    def _compute_svd(self):
        """
        Compute singular value decomposition of the projected genotype matrix.
        """
        if self.VarCompEst:
            self.U, self.S, _ = _numpy_svd(self.Z, compute_uv=True)
        else:
            self.S = _numpy_svd(self.Z)
        self.S = np.square(self.S)
        self.S[self.S <= 1e-6] = 0
        self.filtered = np.nonzero(self.S)[0]
        self.S = self.S[self.filtered]
        if self.VarCompEst:
            self.U = self.U[:, self.filtered]

    def _estimate_variance_components(self):
        """
        Estimate variance components.
        """
        if self.VarCompEst:
            if self.X is None:
                var_est = VarComponentEst(self.S, self.U, self.y)
            else:
                yt = self.U.T @ self.y
                if self.varCompStd:
                    var_est = VarComponentEst_Cov_std(
                        self.S, yt, self.y1, self.y)
                else:
                    var_est = VarComponentEst_Cov(self.S, yt, self.y1, self.y)
            self.sigma2_gxg = var_est[1]
            self.sigma2_e = var_est[2]
            self.trace = np.sum(self.S)  # compute the trace of phi phi.T
            # compute the sum(Phi Phi.T)
            self.sumK = np.sum(np.sum(self.Z, axis=0)**2)
            self.results['varcomp'] = var_est

    def _compute_p_value(self):
        """
        Compute the p-value.
        """
        P1 = self.P1
        if self.center:
            sq_sigma_e0 = (self.y.T @ self.y - self.y.T @ self.X @
                           P1 @ (self.X.T @ self.y))[0] / (self.n - self.k)
        else:
            sq_sigma_e0 = self.y.T @ self.y / self.n

        self.p_value1 = score_test2(
            sq_sigma_e0, self.Q, self.S, center=self.center)

        if self.Perm:
            p_list = [self.p_value1]
            for state in range(self.Perm):
                shuff_idx = np.random.RandomState(
                    seed=state).permutation(self.n)
                yperm = (
                    self.y - (self.X @ (P1 @ (self.X.T @ self.y))))[shuff_idx]
                Qperm = np.sum(np.square(yperm.T @ self.Z))
                sq_sigma_e0_perm = (yperm.T @ yperm)[0] / (self.n - self.k)
                p_value1_perm = score_test2(
                    sq_sigma_e0_perm, Qperm, self.S, center=self.center)
                p_list.append(p_value1_perm)
            self.results['pval'] = p_list
        else:
            self.results['pval'] = self.p_value1

    def run(self):
        """
        Run the FastKAST component analysis.
        """
        self.results = {}
        self._preprocess_data()
        self._feature_map()
        self._compute_svd()
        if self.VarCompEst:
            self._estimate_variance_components()
        self._compute_p_value()
        return self.results


class FastKASTComponentMulti:
    """
    Class for performing FastKAST component analysis for multiple traits.

    This class encapsulates the logic for computing the FastKAST component, 
    including variance component estimation and p-value calculation for multiple traits.
    """

    def __init__(self, X, Z, y, theta=False, dtype='quant', center=True,
                 method='Numpy', Perm=10, Test='linear',
                 VarCompEst=False, varCompStd=False):
        """
        Initialize the FastKASTComponentMulti object.

        Args:
            X (np.ndarray): Covariates to be regressed out.
            Z (np.ndarray): Genotype matrix.
            y (np.ndarray): Phenotype matrix (N x K, where K is the number of traits).
            theta (bool, optional): Whether to include theta in the model. Defaults to False.
            dtype (str, optional): Data type. Defaults to 'quant'.
            center (bool, optional): Whether to center the data. Defaults to True.
            method (str, optional): Method for computation. Defaults to 'Numpy'.
            Perm (int, optional): Number of permutations for p-value calculation. Defaults to 10.
            Test (str, optional): Type of test. Defaults to 'nonlinear'.
            VarCompEst (bool, optional): Whether to estimate variance components. Defaults to False.
            varCompStd (bool, optional): Whether to standardize variance components. Defaults to False.
        """
        self.X = X
        self.Z = Z
        self.y = y
        self.theta = theta
        self.dtype = dtype
        self.center = center
        self.method = method
        self.Perm = Perm
        self.Test = Test
        self.VarCompEst = VarCompEst
        self.varCompStd = varCompStd

        self.n = Z.shape[0]
        self.M = Z.shape[1]
        self.K = y.shape[1]  # Number of traits

    def _preprocess_data(self):
        """
        Preprocess the data by handling missing values, centering, and projecting.
        """
        self.nan_num = np.sum(np.isnan(self.y), axis=0)
        print(f'nan_num is {self.nan_num}')
        self.y = np.nan_to_num(self.y)

        if self.center:
            if self.X is None or self.X.size == 0:
                self.X = np.ones((self.n, 1))
            else:
                self.X = np.concatenate((np.ones((self.n, 1)), self.X), axis=1)

        if self.X is None or self.X.size == 0:
            self.k = 0
            self.Q = np.sum(np.square(self.y.T @ self.Z), axis=1)  # K vector
            self.y1 = self.y.copy()
        else:
            self.k = self.X.shape[1]
            P1 = _inverse(self.X)
            self.P1 = P1
            self.Z = _projection(self.Z, self.X, P1)
            self.Q = np.sum(np.square(self.y.T @ self.Z - self.y.T @
                            self.X @ P1 @ self.X.T @ self.Z), axis=1)  # K vector
            B1, _, _ = _numpy_svd(self.X, compute_uv=True)
            self.y1 = B1.T @ self.y

    def _feature_map(self):
        """
        Perform feature transformation depending on the request
        """
        if self.Test == 'linear':
            self.Z = (self.Z) / np.sqrt(self.Z.shape[1])

        else:
            raise NotImplementedError(f"{self.Test} is not implemented")

    def _compute_svd(self):
        """
        Compute singular value decomposition of the projected genotype matrix.
        """
        if self.VarCompEst:
            self.U, self.S, _ = _numpy_svd(self.Z, compute_uv=True)
        else:
            self.S = _numpy_svd(self.Z)
        self.S = np.square(self.S)
        self.S[self.S <= 1e-6] = 0
        self.filtered = np.nonzero(self.S)[0]
        self.S = self.S[self.filtered]
        if self.VarCompEst:
            self.U = self.U[:, self.filtered]

    def _estimate_variance_components(self):
        """
        Estimate variance components. 
        """
        if self.VarCompEst:
            print(f'Var comp for multi-trait version hasnt implemented yet')
            # ... (Implement variance component estimation for multiple traits)

    def _compute_p_value(self):
        """
        Compute the p-value.
        """
        P1 = self.P1
        if self.center:
            yTXPX = self.y.T @ self.X @ P1 @ self.X.T
            sq_sigma_e0_num = np.sum(
                self.y * self.y, axis=0) - np.sum(yTXPX * (self.y.T), axis=1)
            sq_sigma_e0_den = (self.n - self.k - self.nan_num)
            self.sq_sigma_e0 = sq_sigma_e0_num / sq_sigma_e0_den  # K vector
        else:
            self.sq_sigma_e0 = np.sum(
                self.y * self.y, axis=0) / (self.n - self.nan_num)  # K vector

        self.p_vals = score_test_qf(
            self.sq_sigma_e0, self.Q, self.S, center=self.center, multi=True)

        # Note: Permutation testing for multiple traits is not currently implemented.

    def run(self):
        """
        Run the FastKAST component analysis for multiple traits.
        """
        self.results = {}
        self._preprocess_data()
        self._compute_svd()
        if self.VarCompEst:
            self._estimate_variance_components()
        self._compute_p_value()
        return self.results
