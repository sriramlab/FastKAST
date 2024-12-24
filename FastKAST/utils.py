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
