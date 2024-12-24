import os.path
from joblib import dump, load
import glob
import time
import numpy as np
from scipy.stats import qmc
# from numba_stats import norm
from sklearn.impute import SimpleImputer
from scipy.stats import norm
from numba import jit
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import scale
from scipy.stats import chi2
from scipy.optimize import minimize
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






def impute_def(x):
    col_mean = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_mean, inds[1])
    return x


def impute(x):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    x = imp.fit_transform(x)
    return x