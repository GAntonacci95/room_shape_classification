import pandas as p
import multiprocessing as mp
import numpy as np
from pathlib import Path
import os
from framework.data import fs_utils
from framework.data.hashing import Hasher
from functools import partial

# TODO: REFACTOR
DEFAULT_DIR_MODE = 0o777


def __output_fun_dir(base_output_dir: str, fun_name: str) -> str:
    output_fun_dir = "{}/{}".format(base_output_dir, fun_name)
    return output_fun_dir


def __parametrized_output_fun_dir(input_dir: str, fun_params: dict, output_fun_dir: str) -> str:
    myh = Hasher()
    parametrized_output_dir = "{}/{}".format(
        output_fun_dir,
        myh.compute_hash({"input_dir": input_dir, "fun_params": fun_params})
    )
    return parametrized_output_dir


def get_output_dir(input_dir: str, fun: dict, base_output_dir: str) -> str:
    output_fun_dir = __output_fun_dir(base_output_dir, fun["handler"].__name__)
    ret = __parametrized_output_fun_dir(input_dir, fun["params"], output_fun_dir)
    if not Path(ret).is_dir():
        raise NotADirectoryError("Output directory doesn't exist! - {}".format(ret))
    return ret


def get_i_df(input_dir: str) -> p.DataFrame:
    input_file_names = fs_utils.get_dir_filenames(input_dir)
    ret = []
    for input_file_name in input_file_names:
        ret.append({
            "input_dir": input_dir,
            "input_file_name": input_file_name,
            "input_file_path": "{}/{}".format(input_dir, input_file_name)      # ridondante ma comodo
        })
    return p.DataFrame(ret)


def __fun_io_df(input_dir: str, fun: dict, base_output_dir: str) -> p.DataFrame:
    if not Path(base_output_dir).is_dir():
        raise NotADirectoryError("Directory not found!")

    output_fun_dir = __output_fun_dir(base_output_dir, fun["handler"].__name__)
    parametrized_output_fun_dir = __parametrized_output_fun_dir(input_dir, fun["params"], output_fun_dir)
    # se non è presente la dir per questa parametrizzazione la creo
    if not Path(parametrized_output_fun_dir).is_dir():
        os.makedirs(parametrized_output_fun_dir, mode=DEFAULT_DIR_MODE, exist_ok=False)

    ret = []
    if input_dir is None or input_dir == "":
        # FUNCTION IO INFO (fun DataFrame parameter)
        ret.append({
            "input_dir": input_dir,
            # la fun potrebbe avere più file d'uscita dipendentemente dai parametri
            "output_dir": parametrized_output_fun_dir
        })
    elif not Path(input_dir).is_dir():
        raise NotADirectoryError("Input directory not found!")
    else:
        i_df = get_i_df(input_dir)
        if i_df.shape[0] == 0:
            raise NotADirectoryError("Input directory empty!")
        for t in i_df.itertuples():
            # FUNCTION IO INFO (fun DataFrame parameter)
            ret.append({
                "input_dir": t.input_dir,
                # passo il percorso al file per poter parallelizzare sui files!
                "input_file_name": t.input_file_name,
                "input_file_path": t.input_file_path,
                # la fun potrebbe avere più file d'uscita dipendentemente dai parametri
                "output_dir": parametrized_output_fun_dir
            })
    return p.DataFrame(ret)


def __load_fun_experiments(output_fun_dir: str) -> p.DataFrame:
    # json for info, pkl for raw data
    fun_experim = "{}/fun_experiments.json".format(output_fun_dir)
    fun_experim_df: p.DataFrame = p.DataFrame()
    if Path(fun_experim).is_file():
        fun_experim_df = p.read_json(fun_experim)
    return fun_experim_df


def __store_fun_experiments(fun_experim_df: p.DataFrame, output_fun_dir: str) -> None:
    # json for info, pkl for raw data
    fun_experim = "{}/fun_experiments.json".format(output_fun_dir)
    fun_experim_df.to_json(fun_experim)
    return


# TODO: VERIFICARE CHE QUESTO FUNZIONI e L'IGNORE INDEX VADA BENE
def __update_fun_experiments(input_dir: str, fun: dict, base_output_dir: str) -> None:
    output_fun_dir = __output_fun_dir(base_output_dir, fun["handler"].__name__)
    parametrized_output_fun_dir = __parametrized_output_fun_dir(input_dir, fun["params"], output_fun_dir)
    # se non è presente la dir per questa parametrizzazione esplodo
    if not Path(parametrized_output_fun_dir).is_dir():
        raise NotADirectoryError("Parametrized output directory doesn't exist!")

    fun_experim_df = __load_fun_experiments(output_fun_dir)
    if len(fun_experim_df) == 0 or\
            parametrized_output_fun_dir not in fun_experim_df["fun_parametrized_output_dir"].values:
        # TRACE STORAGE
        detail = {
            "input_dir": input_dir,
            "module_name": fun["handler"].__module__,
            "fun_name": fun["handler"].__name__,
            "fun_params": fun["params"],
            "fun_parametrized_output_dir": parametrized_output_fun_dir
        }
        fun_experim_df = fun_experim_df.append(detail, ignore_index=True)
        __store_fun_experiments(fun_experim_df, output_fun_dir)
    return


def compute_sequential(input_dir: str, fun: dict, base_output_dir: str) -> None:
    # le computazioni sequenziali non necessitano per forza di input_dir
    df = __fun_io_df(input_dir, fun, base_output_dir)
    fun["handler"](df, fun["params"])
    __update_fun_experiments(input_dir, fun, base_output_dir)
    return


# nuovo approccio parallelo per la gestione del calcolo di funzioni su file in input: il parallelismo è sui file
# eg input_dir su tutti devo applicare fun.handler
# fun = {handler: function, params: dict}
# handler(subset: p.DataFrame, params: dict)
def compute_parallel(input_dir: str, fun: dict, base_output_dir: str) -> None:
    # le computazioni parallele di fun richiedono per forza un tot di file in input_dir
    if input_dir is None or input_dir == "" or not Path(input_dir).is_dir():
        raise NotADirectoryError("Parallel computation requires an input_dir!")
    df = __fun_io_df(input_dir, fun, base_output_dir)
    # rows shuffling for threads load-balancing
    input_df = df.sample(frac=1)

    n_procs = mp.cpu_count() - 1    # TODO: PENSARE COME ESSERE PIU' EDUCATO
    if df.shape[0] < n_procs:
        n_procs = df.shape[0]
    subsets = np.array_split(input_df, n_procs)
    pool = mp.Pool(n_procs)
    # fun["handler"]
    #       receives: set: p.DataFrame, params: dict
    #       returns nothing
    # pool.map distributes each subset in subsets to each fun thread
    pool.map(partial(fun["handler"], params=fun["params"]), subsets)
    pool.close()
    pool.join()
    __update_fun_experiments(input_dir, fun, base_output_dir)
    return
