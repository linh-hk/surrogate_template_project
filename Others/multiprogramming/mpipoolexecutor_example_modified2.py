#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:35:35 2023

@author: h_k_linh
"""
"""
Compare the speed of primes sequentially vs. using futures.
"""

import sys
import time
import math
try:
    range = xrange
except NameError:
    range = range

try:
    from concurrent.futures import ThreadPoolExecutor
except ImportError:
    ThreadPoolExecutor = None
try:
    from concurrent.futures import ProcessPoolExecutor
except ImportError:
    ProcessPoolExecutor = None

from mpi4py.futures import MPIPoolExecutor

PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    117450548693743,
    993960000099397,
]

def is_prime(n):
    if n % 2 == 0:
        return False
    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True
if __name__ == '__main__':
    name = 'mpi4py'
    sys.stdout.write('%s: ' % name.ljust(11))
    start = time.time()
    with MPIPoolExecutor(2) as executor:
        results = list(executor.map(is_prime, PRIMES))
    sys.stdout.flush()
    sys.stdout.write('%5.2f seconds\n' % (time.time() - start))
    sys.stdout.flush()
    sys.stdout.write(f'{results}')
    sys.stdout.flush()
