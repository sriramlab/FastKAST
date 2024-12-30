import numpy as np


def _dlik(logdelta, *args):
    n, Sii, UTy, LLadd1 = args
    UTy = UTy.flatten()
    delta = np.exp(logdelta)
    sUTy = np.square(UTy)
    if LLadd1 == None:
        LLadd1 = 0
    L1 = 0.5 * n * (np.sum(sUTy / np.square(Sii + delta)) +
                    LLadd1 / np.square(delta))
    L11 = np.sum(sUTy / (Sii + delta)) + LLadd1 / delta
    L2 = 0.5 * (np.sum(1 / (Sii + delta)) + (n - len(Sii)) * (1 / delta))
    der = np.zeros_like(delta)
    der[0] = -L1 / L11 + L2
    return der


def _lik(logdelta, *args):
    if len(args) == 1:
        nargs = args
        (n, Sii, UTy, LLadd1) = nargs[0]
    else:
        (n, Sii, UTy, LLadd1) = args
    UTy = UTy.flatten()
    nulity = max(0, n - len(Sii))
    L1 = (sum(np.log(Sii + np.exp(logdelta))) +
          nulity * logdelta) / 2  # The first part of the log likelihood
    sUTy = np.square(UTy)
    if LLadd1 is None:
        L2 = (n / 2.0) * np.log((sum(sUTy / (Sii + np.exp(logdelta)))))
    else:
        L2 = (n / 2.0) * np.log((sum(sUTy / (Sii + np.exp(logdelta))) +
                                 (LLadd1 / (np.exp(logdelta)))))
    return (L1 + L2)


def _lik_cov(logdelta, *args):
    if len(args) == 1:
        nargs = args
        (n, Sii, yt, LLadd1) = nargs[0]
    else:
        (n, Sii, yt, LLadd1) = args
    yt = yt.flatten()
    nulity = max(0, n - len(Sii))  # N - K - K'
    L1 = (sum(np.log(Sii + np.exp(logdelta))) +
          nulity * logdelta) / 2  # The first part of the log likelihood
    syt = np.square(yt)

    L2 = (n / 2.0) * np.log((sum(syt / (Sii + np.exp(logdelta))) - sum(syt / np.exp(logdelta)) +
                             (LLadd1 / (np.exp(logdelta)))))
    return (L1 + L2)


def _lik2(param, *args):
    if len(args) == 1:
        nargs = args
        (n, Sii, UTy, LLadd1) = nargs[0]
    else:
        (n, Sii, UTy, LLadd1) = args
    logdelta = param[0]
    gamma = param[1]
    # gamma = 0
    UTy = UTy.flatten()
    nulity = max(0, n - len(Sii))
    L1 = (sum(np.log(Sii * np.exp(gamma) + np.exp(logdelta))) +
          nulity * logdelta) / 2  # The first part of the log likelihood
    sUTy = np.square(UTy)
    if LLadd1 is None:
        # print('operation on L2')
        L2 = (n / 2.0) * np.log(
            (sum(sUTy / (Sii * np.exp(gamma) + np.exp(logdelta)))) / n)
    else:
        L2 = (n / 2.0) * np.log(
            (sum(sUTy / (Sii * np.exp(gamma) + np.exp(logdelta))) +
             (LLadd1 / (np.exp(logdelta)))) / n)
    return (L1 + L2)
