import pandas as p
import framework.data.io_relations.files.parallelization as pdf
from functools import partial
import numpy as np
# import gc
import os
from scipy.io import wavfile
from scipy.signal import fftconvolve


# irs_fs is not stored, however it shall be the real one.
# The wav input_files must have the same fs as the irs
def __sub_simulate(
        irss_subset: p.DataFrame,
        irs_fs: int,
        input_base_dir: str
) -> p.DataFrame:
    ret = []
    for irs_tpl in irss_subset.itertuples():
        irs = irs_tpl.irs
        # per ogni tupla convolvo con tutti i file in input_base_dir
        for _, _, input_file_names in os.walk(input_base_dir):
            for input_file_name in [file_name for file_name in input_file_names if ".wav" in file_name]:
                input_file_path = "{}/{}".format(input_base_dir, input_file_name)
                file_fs, signal = wavfile.read(input_file_path)
                if irs_fs != file_fs:   # resampling instead?
                    raise Exception("The irs_fs differs from the file_fs")
                # convoluzione lungo_x rpt(signal, lungo_y) ed irs - check convenzione √
                output_buffer = fftconvolve(np.tile(signal, (len(irs), 1)), irs, axes=1)

                ret.append({
                    "room_id": irs_tpl.room_id,     # ridondante, ma comodo
                    "irs_id": irs_tpl.Index,
                    "input_file_path": input_file_path,
                    "output_buffer": output_buffer
                })
    return p.DataFrame(ret)


def parallel_simulate(
        irss: p.DataFrame,
        irs_fs: int,
        input_base_dir: str
):
    return pdf.compute_parallel(
        irss,
        partial(__sub_simulate, irs_fs=irs_fs,
                input_base_dir=input_base_dir)
    )


def simulate(
        irss: p.DataFrame,
        irs_fs: int,
        input_base_dir: str
) -> p.DataFrame:
    return __sub_simulate(irss, irs_fs, input_base_dir)
