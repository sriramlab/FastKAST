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


@jit(nopython=True)
def sin_cos(X, method='sin'):
    if method == 'sin':
        results = np.sin(X)
    else:
        results = np.cos(X)
    return results


def CCT(pvals, ws=None):
    N = len(pvals)
    if not ws:
        ws = np.array([1 / N for i in range(N)])
    T = np.sum(ws * np.tan((0.5 - pvals) * np.pi))
    pval = 0.5 - np.arctan(T) / np.pi
    return pval



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
    
    
def mix_chi_fit(weights, x, phi, topk):
    '''
    Used for LRT -- fitting the mixture of chi-square given phi/pi
    '''
    samples = len(x)
    a, d = weights
    
    topk_samples = int(topk*samples)
    quantiles = np.linspace(0, 1, samples)[-topk_samples:] - 0.5/samples
    
    # theoretical_dist = mix_chi_quantile(quantiles,phi, a, d)
    
    top_empirical = np.sort(x)[-topk_samples:]
    # print(f'empirical: {top_empirical}')
    
    top_theoretical = np.array([mix_chi_quantile(q, phi, a, d) for q in quantiles])
    # print(f'theoretical: {top_theoretical}')

    loss = np.sum(np.square(np.log(top_empirical+1e-10) - np.log(top_theoretical+1e-10)))
    # print(f'loss is {loss}')
    return loss

def mix_chi_quantile(q, phi, a, d):
    """
    Compute the quantile for the mixture of chi-square distributions.
    Args:
        q: Quantile (e.g., 0.9 for the 90th percentile).
        phi: Proportion of zeros in the mixture.
        a: Scaling factor for chi(d).
        d: Degrees of freedom for chi(d).
    Returns:
        Theoretical quantile value.
    """
    # Adjust for the mixture
    if q <= phi:
        return 0  # Quantiles below phi correspond to chi(0)
    else:
        q_adj = (q - phi) / (1 - phi)
        return a * chi2.ppf(q_adj, d)
    
    
def fit_null(x,top_ratio=0.1):
    '''
    Using LRT permutation statistics to perform curve fitting
    Args:
        x: log LR statistics
        top_ratio: the top ratio used for curve fitting
    '''
    x = np.array(x)
    bounds = [(1e-2, 100), (1e-2, 100)]
    top=int(top_ratio*len(x))
    phi=np.mean(x<=1e-6)
    init_x = [1e0,1e0]
    result = minimize(mix_chi_fit, init_x, args=(np.sort(x), phi, top_ratio), bounds=bounds, method="L-BFGS-B",options={'gtol': 1e-6, 'maxiter':1e5})
    return result, phi


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
