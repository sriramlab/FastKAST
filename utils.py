import os.path
import joblib
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

def resumefile(path, filename):
    filepath=f'{path}{filename}'
    data = joblib.load(filepath)
    return data

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


@jit(nopython=True)
def sin_cos(X, method='sin'):
    if method == 'sin':
        results = np.sin(X)
    else:
        results = np.cos(X)
    return results


class QMC_RFF:
    def __init__(self, gamma, d, n_components, seed=None, QMC='Halton'):
        assert n_components % 2 == 0
        self.gamma = gamma
        self.n_components = n_components
        self.seed = seed
        self.QMC = QMC
        if QMC == 'Halton':
            sampler = qmc.Halton(d=d, seed=seed)
            self.sampler = sampler
        elif QMC == 'Sobol':
            sampler = qmc.Sobol(d=d, seed=seed)
            self.sampler = sampler
        else:
            raise ValueError(f"{QMC} is currently not supported")

    # @jit(nopython=True)
    def fit_transform(self, X):
        n_components = self.n_components
        sampler = self.sampler
        ts = sampler.random(n=n_components // 2)
        gamma = self.gamma
        # t0 = time.time()
        W = norm.ppf(ts, scale=np.sqrt(2 * gamma)).T
        # t1 = time.time()
        # print(f'get W takes {t1-t0}')
        self.W = W
        projection = X @ W
        # t0 = time.time()
        sin = np.sin(projection)
        cos = np.cos(projection)
        # sin = sin_cos(projection)
        # cos = sin_cos(projection,method='cos')
        # t1 = time.time()
        # print(f'sin cos takes {t1-t0}')
        Combine = np.empty((sin.shape[0], 2 * sin.shape[1]), dtype=float)
        Combine[:, 0::2] = sin
        Combine[:, 1::2] = cos
        # t1 = time.time()
        # print(f'assign takes {t1-t0}')
        # t0 = time.time()
        Combine *= np.sqrt(2.) / np.sqrt(n_components)
        # t1 = time.time()
        # print(f'Combin mult takes {t1-t0}')
        return np.float32(Combine)


if __name__ == "__main__":
    N = 1000
    M = 10
    D = 50
    p = np.random.uniform(size=M)
    X = scale(np.random.binomial(n=2, p=p, size=(N, M)))
    gamma = 0.1
    K = rbf_kernel(X, gamma=gamma)
    distances = []
    for i in range(10):
        Z = RBFSampler(gamma=gamma, n_components=D * M,
                       random_state=i).fit_transform(X)

        distance = np.linalg.norm(Z @ Z.T - K, 'fro')
        distances.append(distance)
    distances = np.array(distances)
    print('standard RFF approximation loss:', np.mean(distances),
          np.std(distances))

    distances = []
    for i in range(10):
        Z1 = QMC_RFF(gamma=gamma,
                     d=M,
                     n_components=D * M,
                     QMC='Halton',
                     seed=i)
        Z1 = Z1.fit_transform(X)
        distance = np.linalg.norm(Z1 @ Z1.T - K, 'fro')
        distances.append(distance)
    distances = np.array(distances)
    print('QMC RFF approximation loss ', np.mean(distances), np.std(distances))

    # path = './test/'
    # data = ['a']
    # filename = 'testfile.pkl'
    # dumpfile(data,path,filename,overwrite=True)
