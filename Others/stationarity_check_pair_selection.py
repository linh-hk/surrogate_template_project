# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 00:40:39 2026

@author: hoang
"""
from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np

def adf_p(x, regression="c", autolag="AIC", maxlag=None):
    x = np.asarray(x, dtype=float)
    return adfuller(x, regression=regression, autolag=autolag, maxlag=maxlag)[1]

def kpss_p(x, regression="c", nlags="auto"):
    x = np.asarray(x, dtype=float)
    return kpss(x, regression=regression, nlags=nlags)[1]

def stationary_evaluate(X, alpha=0.05, regression="c", 
                        autolag="AIC", maxlag=None, 
                        nlags="auto"):
    """
    X: (T,S)
    Returns boolean mask (S,) for species that are stationary over the whole window:
      ADF p < alpha  AND  KPSS p > alpha
    """
    X = np.asarray(X, dtype=float)
    S = X.shape[1]
    mask = np.zeros(S, dtype=bool)

    for s in range(S):
        x = X[:, s]
        p_adf = adf_p(x, regression=regression, autolag=autolag, maxlag=maxlag)
        p_kpss = kpss_p(x, regression=regression, nlags=nlags)
        mask[s] = (p_adf < alpha) and (p_kpss > alpha)

    return mask

def choose_pair(trial, alpha=0.05, top_k=2, value="mean", abs_value=False,
                regression="c", autolag="AIC", maxlag=None, nlags="auto"):
    """
    Filters by stationarity over the full window using:
      ADF p < alpha AND KPSS p > alpha

    Returns:
      pair_ts   : (T,2) array
      pair_idx  : (s1,s2) original species indices
      stationary: array of stationary species indices (for trace/debug)
    """
    mask = stationary_evaluate(
        trial, alpha=alpha, regression=regression, autolag=autolag,
        maxlag=maxlag, nlags=nlags
    )
    stationary = np.where(mask)[0]
    if stationary.size < 2:
        return None, (-1, -1), stationary

    # value to rank (not returned)
    if value == "median":
        vals = np.median(trial, axis=0)
    else:
        vals = np.mean(trial, axis=0)

    key = np.abs(vals[stationary]) if abs_value else vals[stationary]
    order = stationary[np.argsort(key)[::-1]]

    s1 = int(order[0])
    s2 = int(order[min(top_k - 1, order.size - 1)])

    pair_ts = trial[:, [s1, s2]]
    return pair_ts, (s1, s2), stationary

def refine_run_data(data, N0,
                    alpha=0.05, top_k=2, value="mean", abs_value=False):
    pairs = []
    series_ids = []
    species_id = []
    stationary_idx = []

    for i, trial in enumerate(data):
        pair, (s1, s2), stationarity_ = choose_pair(trial, alpha=alpha, top_k=top_k, value=value, abs_value=abs_value)
        if pair is None:
            continue
        pairs.append((pair, N0+i)) 
        series_ids.append(N0+i)
        species_id.append((s1, s2))
        stationary_idx.append(stationarity_)

    meta = {"series_ids": series_ids, "species_id": species_id, "stationary_idx": stationary_idx}

    return pairs, meta

# test, test_meta = refine_run_data(data['data'][:100], N0=0, top_k=2)
# for series in test:
#     plot_timeseries_all_species(series[0], title = "top2 both")
# pick top 2 because after looking, it seems that most of them are close to 0 after filtered

alpha = 0.05
top_k = 2
value = "mean"
abs_value = False
data, meta = refine_run_data(data=data, N0=N0,
                             alpha=alpha, top_k=top_k, value=value, abs_value=abs_value)
data = falsepos_datagen(data)