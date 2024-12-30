import numpy as np
# from numba_stats import norm
from scipy.stats import chi2
from scipy.optimize import minimize
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import fastlmmclib.quadform as qf
from chi2comb import chi2comb_cdf, ChiSquared


############### p-value aggregation ###############

def CCT(pvals, ws=None):
    N = len(pvals)
    if not ws:
        ws = np.array([1 / N for i in range(N)])
    T = np.sum(ws * np.tan((0.5 - pvals) * np.pi))
    pval = 0.5 - np.arctan(T) / np.pi
    return pval


############### Score Testing ###############

def score_test(S):
    # N = Z.shape[0]
    k = len(S)
    dofs = np.ones(k)
    ncents = np.zeros(k)
    chi2s = [ChiSquared(S[i], ncents[i], dofs[i]) for i in range(k)]
    p, error, info = chi2comb_cdf(0, chi2s, 0, lim=10000000, atol=1e-14)
    # p = qf.qf(0, Phi, acc = 1e-7)[0]
    return (1 - p, error)


def score_test2(sq_sigma_e0, Q, S, decompose=True, center=False, multi=False, DEBUG=False):
    k = len(S)
    Phi = np.zeros(k)
    Phi[0:len(S)] = S
    Qe = Q / (sq_sigma_e0)
    dofs = np.ones(k)
    ncents = np.zeros(k)
    chi2s = [ChiSquared(Phi[i], ncents[i], dofs[i]) for i in range(k)]
    # t0 = time.time()
    if multi:
        ps = []
        errors = []
        infos = []
        for K in tqdm(range(len(Qe)), desc="Processing score statistics"):
            p, error, info = chi2comb_cdf(
                Qe[K], chi2s, 0, lim=int(1e8), atol=1e-13)
            ps.append(p)
            errors.append(error)
            infos.append(info)
        if DEBUG:
            print(infos)
        ps = np.array(ps)
        return (1-ps, errors)
    else:
        p, error, info = chi2comb_cdf(Qe, chi2s, 0, lim=int(1e8), atol=1e-13)
        if DEBUG:
            print(info)
        # p = qf.qf(0, Phi, acc = 1e-7)[0]
        # t1 = time.time()
        return (1 - p, error)


def score_test_qf(sq_sigma_e0, Q, S, decompose=True, center=False, multi=False):
    Qe = (Q / (sq_sigma_e0))
    if multi:
        ps = []
        for K in tqdm(range(len(Qe)), desc="Processing score statistics"):
            stats = qf.qf(Qe[K], S, sigma=1, lim=int(1e8), acc=1e-15)
            p = stats[0]
            ps.append(p)
        ps = np.array(ps)
        return (ps)
    else:
        stats = qf.qf(Qe, S, sigma=1, lim=int(1e8), acc=1e-15)
        p = stats[0]
        return (p)


############### Likelihood Ratio Testing ###############

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

    top_theoretical = np.array(
        [mix_chi_quantile(q, phi, a, d) for q in quantiles])
    # print(f'theoretical: {top_theoretical}')

    loss = np.sum(np.square(np.log(top_empirical+1e-10) -
                  np.log(top_theoretical+1e-10)))
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


def fit_null(x, top_ratio=0.1):
    '''
    Using LRT permutation statistics to perform curve fitting
    Args:
        x: log LR statistics
        top_ratio: the top ratio used for curve fitting
    '''
    x = np.array(x)
    bounds = [(1e-2, 100), (1e-2, 100)]
    top = int(top_ratio*len(x))
    phi = np.mean(x <= 1e-6)
    init_x = [1e0, 1e0]
    result = minimize(mix_chi_fit, init_x, args=(np.sort(x), phi, top_ratio),
                      bounds=bounds, method="L-BFGS-B", options={'gtol': 1e-6, 'maxiter': 1e5})
    return result, phi
