import pandas as p
import multiprocessing as mp
import numpy as np
from functools import partial


def compute_parallel(df: p.DataFrame, handler, params) -> p.DataFrame:
    n_procs = mp.cpu_count() - 1
    # rows shuffling for load-balancing
    df2 = df.sample(frac=1)
    subsets = np.array_split(df2, n_procs)
    pool = mp.Pool(n_procs)
    ret = p.concat(pool.map(partial(handler, params=params), subsets), ignore_index=True)
    pool.close()
    pool.join()
    return ret
