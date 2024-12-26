import numpy as np
# from numba_stats import norm
from sklearn.impute import SimpleImputer
import numpy as np






def impute_def(x):
    col_mean = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_mean, inds[1])
    return x


def impute(x):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    x = imp.fit_transform(x)
    return x