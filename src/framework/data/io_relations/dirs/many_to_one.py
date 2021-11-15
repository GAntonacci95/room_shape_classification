import pandas as p
import multiprocessing as mp
import numpy as np
from pathlib import Path
import os
from framework.data.hashing import Hasher
from functools import partial
from typing import List

import framework.data.io_relations.dirs.one_to_one_or_many as exp1
from framework.data import fs_utils

# TODO: REFACTOR
def __parametrized_output_fun_dir(input_dirs: List[str], fun_params: dict, output_fun_dir: str) -> str:
    myh = Hasher()
    parametrized_output_dir = "{}/{}".format(
        output_fun_dir,
        myh.compute_hash({"input_dirs": input_dirs, "fun_params": fun_params})
    )
    return parametrized_output_dir


def get_output_dir(input_dirs: List[str], fun: dict, base_output_dir: str) -> str:
    output_fun_dir = exp1.__output_fun_dir(base_output_dir, fun["handler"].__name__)
    ret = __parametrized_output_fun_dir(input_dirs, fun["params"], output_fun_dir)
    if not Path(ret).is_dir():
        raise NotADirectoryError("Directory doesn't exist!")
    return ret


def __fun_io_df(input_dirs: List[str], fun: dict, base_output_dir: str) -> p.DataFrame:
    if not Path(base_output_dir).is_dir():
        raise NotADirectoryError("Directory not found!")

    output_fun_dir = exp1.__output_fun_dir(base_output_dir, fun["handler"].__name__)
    parametrized_output_fun_dir = __parametrized_output_fun_dir(input_dirs, fun["params"], output_fun_dir)
    # se non è presente la dir per questa parametrizzazione la creo
    if not Path(parametrized_output_fun_dir).is_dir():
        os.makedirs(parametrized_output_fun_dir, mode=exp1.DEFAULT_DIR_MODE, exist_ok=False)

    ret = []

    # THE ASSUMPTION HERE IS THAT ALL THE DIRS DO CONTAIN FILES WITH THE SAME NAME
    input_file_names = fs_utils.get_dir_filenames_unique_part(input_dirs[0])
    if len(input_file_names) == 0:
        raise NotADirectoryError("Directory empty!")
    for input_file_name in input_file_names:
        ret.append({
            "input_dirs": input_dirs,
            "input_file_name": input_file_name,
            "output_dir": parametrized_output_fun_dir
        })
    return p.DataFrame(ret)


# TODO: VERIFICARE CHE QUESTO FUNZIONI e L'IGNORE INDEX VADA BENE
def __update_fun_experiments(input_dirs: List[str], fun: dict, base_output_dir: str) -> None:
    output_fun_dir = exp1.__output_fun_dir(base_output_dir, fun["handler"].__name__)
    parametrized_output_fun_dir = __parametrized_output_fun_dir(input_dirs, fun["params"], output_fun_dir)
    # se non è presente la dir per questa parametrizzazione esplodo
    if not Path(parametrized_output_fun_dir).is_dir():
        raise NotADirectoryError("Directory doesn't exist!")

    fun_experim_df = exp1.__load_fun_experiments(output_fun_dir)
    if len(fun_experim_df) == 0 or\
            parametrized_output_fun_dir not in fun_experim_df["fun_parametrized_output_dir"].values:
        # TRACE STORAGE
        detail = {
            "input_dirs": input_dirs,
            "module_name": fun["handler"].__module__,
            "fun_name": fun["handler"].__name__,
            "fun_params": fun["params"],
            "fun_parametrized_output_dir": parametrized_output_fun_dir
        }
        fun_experim_df = fun_experim_df.append(detail, ignore_index=True)
        exp1.__store_fun_experiments(fun_experim_df, output_fun_dir)
    return


def compute_sequential(input_dirs: List[str], fun: dict, base_output_dir: str) -> None:
    if input_dirs is None or len(input_dirs) == 0 or any(not Path(input_dir).is_dir() for input_dir in input_dirs):
        raise NotADirectoryError("Computation requires valid input_dirs!")
    df = __fun_io_df(input_dirs, fun, base_output_dir)
    fun["handler"](df, fun["params"])
    __update_fun_experiments(input_dirs, fun, base_output_dir)
    return


def compute_parallel(input_dirs: List[str], fun: dict, base_output_dir: str) -> None:
    if input_dirs is None or len(input_dirs) == 0 or any(not Path(input_dir).is_dir() for input_dir in input_dirs):
        raise NotADirectoryError("Computation requires valid input_dirs!")
    df = __fun_io_df(input_dirs, fun, base_output_dir)
    # rows shuffling for threads load-balancing
    input_df = df.sample(frac=1)

    n_procs = mp.cpu_count() - 1    # TODO: PENSARE COME ESSERE PIU' EDUCATO
    subsets = np.array_split(input_df, n_procs)
    pool = mp.Pool(n_procs)

    pool.map(partial(fun["handler"], params=fun["params"]), subsets)
    pool.close()
    pool.join()
    __update_fun_experiments(input_dirs, fun, base_output_dir)
    return
