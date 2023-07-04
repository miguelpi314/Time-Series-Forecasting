# -*- coding: utf-8 -*-
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
import numpy as np


def stationary_test(data, alpha=0.05):
    ADF_result = adfuller(data)
    return ADF_result[1] < alpha


def autocorrelation_test(data, alpha=0.05):
    acf_values, confidence_intervals = acf(data, alpha=alpha)
    return all([(val >= interval[0]) and (val <= interval[1]) for val, interval in
                zip(acf_values[1:], confidence_intervals[1:])])


def is_random_walk(random_walk_array):
    fod = np.diff(random_walk_array, n=1)
    return stationary_test(fod) and autocorrelation_test(fod)
