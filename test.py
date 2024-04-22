import multiprocessing as mp
import math
import time

def fibonacci(n):
    if n <= 1:
        return n
    fib_prev = 0
    fib_current = 1
    for _ in range(2, n + 1):
        fib_prev, fib_current = fib_current, fib_prev + fib_current
    return fib_current


def multiprocessing_fibonacci(function_call: int):
    context = mp.get_context('fork')
    with context.Pool() as p:
        list(p.map(fibonacci, [30000]*function_call, chunksize=1))

if __name__ == '__main__':

    t0 = time.time()
    multiprocessing_fibonacci(300)
    t1 = time.time()
    print(f'{t1-t0:.2e}')