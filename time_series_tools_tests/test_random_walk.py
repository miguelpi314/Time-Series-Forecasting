# -*- coding: utf-8 -*-
from time_series_tools.random_walk import stationary_test, autocorrelation_test, is_random_walk
import numpy as np


def test_random_walk():
    np.random.seed(1)
    initial_value = 1
    random_walk = np.zeros(1000)
    random_walk[0] = initial_value
    for i in range(1, 1000):
        random_walk[i] = random_walk[i - 1] + np.random.normal()
    fod = np.diff(random_walk, n=1)
    assert stationary_test(fod)
    assert autocorrelation_test(fod)
    assert is_random_walk(fod)
