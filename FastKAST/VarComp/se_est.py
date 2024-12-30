import numpy as np
# from numba_stats import norm
import numpy as np


def standerr(U, y, Sii, UTy, g, e):
    L11 = np.sum(np.square(UTy.flatten()) * np.square(Sii) / (g * Sii + e)**3)
    L22 = np.sum(np.square(y - U @ UTy).flatten() / (e)**3) + np.sum(
        np.square(UTy).flatten() / ((e + Sii * g)**3))
    L12 = np.sum((np.square(UTy.flatten()) * Sii) / (g * Sii + e)**3)
    L = 0.5 * np.array([[L11, L12], [L12, L22]])
    cov = np.linalg.inv(L)
    gerr = np.sqrt(cov[0][0])
    eerr = np.sqrt(cov[1][1])
    return [gerr, eerr]



def standerr_dev_cov(Sii, yt, LLadd1, n, g, e):
    '''
    This is the default standard deviation method to use
    '''
    nulity = max(0, n - len(Sii))
    # print(f'nulity is {nulity}')
    # print(f'Sii: {Sii}')
    # print(f'g: {g}')
    # print(f'e: {e}')
    L11 = -0.5*(np.sum(np.square(Sii) / np.square(g * Sii + e))) + \
        np.sum(np.square(yt.flatten()) * np.square(Sii) / (g * Sii + e)**3)
    L22 = -0.5*(np.sum(1. / np.square(g * Sii + e))+nulity*1./e**2) + np.sum(np.square(
        yt.flatten()) / (g * Sii + e)**3) - np.sum(np.square(yt.flatten())/e**3) + LLadd1 / e**3
    L12 = -0.5*np.sum(Sii / np.square(g * Sii + e)) + \
        np.sum(np.square(yt.flatten()) * Sii / (g * Sii + e)**3)
    # print(np.sum(np.square(UTy.flatten()) * Sii / (g * Sii + e)**3))
    # print(-0.5*np.sum(Sii/(g * Sii + e)**2))
    L = np.array([[L11, L12], [L12, L22]])
    # print(f'L is:')
    # print(L)
    cov = np.linalg.inv(L)
    # print(f'Cov is:')
    # print(cov)
    gerr = np.sqrt(cov[0][0])
    eerr = np.sqrt(cov[1][1])
    return [gerr, eerr]
