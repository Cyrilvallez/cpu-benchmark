import time
import random

import numpy as np
from scipy.spatial.distance import pdist

# Set the random seeds
random.seed(0)
np.random.seed(0)

N = 10


def timeit(N_repeat: int = 10):
    def decorator(decorated_function):
        def wrapper(*args, **kwargs):

            times = []
            for _ in range(N_repeat):
                t0 = time.time()
                result = decorated_function(*args, **kwargs)
                t1 = time.time()
                times.append(t1 - t0)

            timings = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times)
            }
            print('    ' + ' | '.join(f'{k}: {v:.3e}' for k, v in timings.items()))

            return result
        return wrapper
    return decorator


@timeit(N)
def matrix_multiplication(dim: int):

    a = np.random.rand(dim, dim)
    b = np.random.rand(dim, dim)

    return a@b


@timeit(N)
def pairwise_euclidean_distance(dim: int):

    X = np.random.rand(dim, 2048)

    return pdist(X, metric='euclidean')



if __name__ == '__main__':

    for dim in (500, 1000, 2000, 5000):
        print(f'Matrix multiplication with dimension {dim}:')
        matrix_multiplication(dim)


    for dim in (500, 1000, 2000, 5000):
        print(f'Pairwise euclidean distance with dimension {dim}:')
        pairwise_euclidean_distance(dim)