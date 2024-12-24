import os.path
from joblib import dump, load
import glob
import time
import numpy as np
from scipy.stats import qmc
# from numba_stats import norm
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






def dumpfile(data, path, filename, overwrite=False):
    isdir = os.path.isdir(path)
    if not isdir:
        os.makedirs(path)
    filepath = f'{path}{filename}'
    isfile = os.path.isfile(filepath)
    if isfile:
        if overwrite:
            print(f'overwrite the existing file')
            dump(data, filepath)
        else:
            print(f'file exists, please enforce overwrite to overwriet it')
    else:
        dump(data, filepath)


def readfiles(rootPath, patternFile):
    traits = []
    for name in glob.glob(f'{rootPath}{patternFile}'):
        traits.append(name)
    return traits


def fileExist(path, filename):
    isdir = os.path.isdir(path)
    if not isdir:
        return False
    filepath = f'{path}{filename}'
    isfile = os.path.isfile(filepath)
    if not isfile:
        return False
    return True

