import numpy as np
import warnings

def test_independence(x, y, w, statistic, x_lag=0):
    """Test for independence between two time series x and y
    Args:
        x (1d numpy array): truncated x is not shifted in time to produce surrogates.
        y (1d numpy array): truncated y is shifted in time to produce surrogates;
            y must be same length as x.
        w (int): w is the width; the truncated time series are of length n-2w,
            where n is the length of the original time series
        statistic (function): a function with the following signature:

            results = statistic(x_trunc, Y_TRUNC)

            here x_trunc is a 1d array of length m and Y_TRUNC is an m by k
            array, where k is the number of shifts. That is, the shifted y_trunc
            time series are the columns of Y_TRUNC. results is a 1d numpy array
            such that results[k] is equal to the statistic performed between
            x_trunc and the kth column of Y_TRUNC
        x_lag (int): the (fixed) shift of the truncated x time series; x_trunc
            begins at measurement w+x_lag.
    Returns:
        b: number of time shifts (including zero shift) with as large a statistc
            value as the zero shift
        p: under the null hypothesis, p is the upper bound on the probability
            that at most b time shifts have as large a statistic value as the
            zero shift
        shift: an array of time shifts
        distribution: distribution[i] is the value of the statistic at shift[i]
    """
    # check user input
    if x.size != y.size:
        raise ValueError('x and y must be the same size')
    n = x.size
    m = n - 2*w
    if m < 1:
        warnings.warn('w is too big; returning NaN')
        return [np.nan, np.nan, np.array([np.nan]), np.array([np.nan])]

    # compute shifted statistics
    x_trunc = x[w+x_lag:w+m+x_lag]
    shift = np.concatenate([np.arange(0,w+1), -np.arange(1,w+1)])
    Y_TRUNC = np.zeros([m, shift.size])
    for i in range(shift.size):
        Y_TRUNC[:,i] = y[w+shift[i]:w+m+shift[i]]
    distribution = statistic(x_trunc, Y_TRUNC)

    # compute p-value
    b = np.sum(distribution >= distribution[0])
    p = b / (w + 1)
    return b, p, shift, distribution

def pcorr(x,y):
    M = np.zeros([x.size, y.shape[1] + 1])
    M[:,0] = x
    M[:,1:] = y
    cov = np.cov(M.T)
    cov_xy = cov[0,:]
    std_y = np.sqrt(np.diag(cov))
    std_x = std_y[0]
    rho = cov_xy / (std_y * std_x)
    return rho[1:]

def pcorr_strength(x,y):
    return np.abs(pcorr(x,y))
