import pandas as p
import framework.data.io_relations.files.parallelization as pdf
from functools import partial
import numpy as np
# import gc
from scipy.signal import fftconvolve


# irs_fs is not stored, however it shall be the real one.
# The wav input_files must have the same fs as the irs
def __sub_simulate(
        irss_subset: p.DataFrame,
        irs_fs: int,
        base_dir: str
) -> p.DataFrame:
    ret = []
    input_dir = "{}/input".format(base_dir)
    output_dir = "{}/output".format(base_dir)

    for irs_tpl in irss_subset.itertuples():
        irs = irs_tpl.irs
        toformat = "{}/wnoise_0.3s_{}.npy"
        input_file_path = toformat.format(input_dir, irs_tpl.Index)
        output_file_path = toformat.format(output_dir, irs_tpl.Index)

        wnoise = np.random.randn(int(irs_fs * 0.3))
        # convoluzione lungo_x rpt(wnoise, lungo_y) ed irs - check convenzione √
        output_buffer = fftconvolve(np.tile(wnoise, (len(irs), 1)), irs, axes=1)

        np.save(input_file_path, wnoise)
        np.save(output_file_path, output_buffer)

        ret.append({
            "input_file_path": input_file_path,
            "room_id": irs_tpl.room_id,  # ridondante, ma comodo
            "irs_id": irs_tpl.Index,
            "output_file_path": output_file_path    # ci sta la gestione con hashing...
        })
    return p.DataFrame(ret)


def parallel_simulate(
        irss: p.DataFrame,
        irs_fs: int,
        base_dir: str
):
    return pdf.compute_parallel(
        irss,
        partial(__sub_simulate, irs_fs=irs_fs,
                base_dir=base_dir)
    )


def simulate(
        irss: p.DataFrame,
        irs_fs: int,
        base_dir: str
) -> p.DataFrame:
    return __sub_simulate(irss, irs_fs, base_dir)
