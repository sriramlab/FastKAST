import numpy as np
from numba import jit
from scipy import stats

@jit(nopython=True)
def sin_cos(X, method='sin'):
    if method == 'sin':
        results = np.sin(X)
    else:
        results = np.cos(X)
    return results


def direct_self(geno_matrix_in):
    N = geno_matrix_in.shape[0]
    M = geno_matrix_in.shape[1]
    D = int((M*(M+1))/2)
    exact = np.zeros((N, D))
    s = 0
    for i in range(M):
        for j in range(i, M):
            feature = geno_matrix_in[:, i]*geno_matrix_in[:, j]
            exact[:, s] = feature
            s += 1
    exact_standard = stats.zscore(exact)

    return exact_standard


def direct_noself(geno_matrix_in):
    N = geno_matrix_in.shape[0]
    M = geno_matrix_in.shape[1]
    D = int((M*(M-1))/2)
    exact = np.zeros((N, D))
    s = 0
    for i in range(M):
        for j in range(i+1, M):
            feature = geno_matrix_in[:, i]*geno_matrix_in[:, j]
            exact[:, s] = feature
            s += 1
    exact_standard = stats.zscore(exact)

    return exact_standard


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
        Combine = np.empty((sin.shape[0], 2 * sin.shape[1]), dtype=float)
        Combine[:, 0::2] = sin
        Combine[:, 1::2] = cos
        Combine *= np.sqrt(2.) / np.sqrt(n_components)
        # t1 = time.time()
        # print(f'Combin mult takes {t1-t0}')
        return np.float32(Combine)
