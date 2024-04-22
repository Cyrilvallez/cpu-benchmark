import time
import random
import math
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

# # Set the random seeds
random.seed(0)
np.random.seed(0)

N = 10

TIMING_DF = pd.DataFrame()

def timeit(N_repeat: int = 10, label: str | None = None):
    def decorator(decorated_function):
        def wrapper(*args, **kwargs):

            # Executes the function N_repeat times
            times = []
            for _ in range(N_repeat):
                t0 = time.time()
                result = decorated_function(*args, **kwargs)
                t1 = time.time()
                times.append(t1 - t0)

            # Compute timing stats
            timings = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times)
            }

            # Create label (name)
            nonlocal label
            if label is None:
                label = decorated_function.__name__

            # Print stats
            print(f'{label} with size {args[0]}:')
            print('    ' + ' | '.join(f'{k}: {v:.2e}' for k, v in timings.items()))

            results = {
                'label': label,
                'size': args[0],
                **timings
            }

            # Add stats to global dataframe for later use
            global TIMING_DF
            TIMING_DF = pd.concat([TIMING_DF, pd.DataFrame.from_records([results])], ignore_index=True)

            return result
        return wrapper
    return decorator


def fibonacci(n):
    if n <= 1:
        return n
    fib_prev = 0
    fib_current = 1
    for _ in range(2, n + 1):
        fib_prev, fib_current = fib_current, fib_prev + fib_current
    return fib_current



@timeit(N)
def numpy_matrix_multiplication(dim: int):
    a = np.random.rand(dim, dim)
    b = np.random.rand(dim, dim)
    return a@b


@timeit(N)
def numpy_matrix_inversion(dim: int):
    a = np.random.rand(dim, dim)
    return np.linalg.inv(a)


@timeit(N)
def scipy_pairwise_euclidean_distance(dim: int):
    X = np.random.rand(dim, 2048)
    return pdist(X, metric='euclidean')


@timeit(N)
def numpy_linear_fit(dim: int):
    x = np.linspace(-10, 10, dim)
    y = 2*x**2 + 4 + np.random.rand(len(x))
    return np.polynomial.Polynomial.fit(x, y, deg=2)


@timeit(10)
def multiprocessing_fibonacci(function_call: int):
    context = mp.get_context('fork')
    with context.Pool() as p:
        list(p.map(fibonacci, [30000]*function_call, chunksize=1))



if __name__ == '__main__':

    DIMS = (500, 1000, 2000, 5000)
    # DIMS = (500, 1000)

    for dim in DIMS:
        numpy_matrix_multiplication(dim)

    for dim in DIMS:
        numpy_matrix_inversion(dim)

    for dim in DIMS:
        scipy_pairwise_euclidean_distance(dim)

    for dim in (5000, 10000, 20000, 30000):
    # for dim in (100, 500):
        numpy_linear_fit(dim)

    for dim in (50, 100, 200, 300):
        multiprocessing_fibonacci(dim)


    grouped = TIMING_DF.set_index(['label', 'size'])
    # print(grouped)
    print(grouped.to_string(index_names=False, col_space=6, justify='center', float_format=lambda x: f'{x:.2e}'))
    print(grouped.to_latex(index_names=False, column_format='lccccc', escape=True, float_format=lambda x: f'{x:.2e}'))