from framework.data import fs_utils
import pandas as p


def get_len_max_of_regions_in_dir(dir: str) -> int:
    maxL = -1
    fdrfiles = fs_utils.get_dir_filepaths(dir)
    for i in range(len(fdrfiles)):
        fdr = p.read_pickle(fdrfiles[i]).output.values[0]

        l = fdr["end_at"] - fdr["begin_at"] + 1
        if l > maxL:
            maxL = l
    return maxL


def remove_fdrs_out_of_range(dir: str, len_min: int, len_max: int):
    import os
    fdrfiles = fs_utils.get_dir_filepaths(dir)
    for i in range(len(fdrfiles)):
        mayrem = fdrfiles[i]
        maydf = p.read_pickle(mayrem)
        if not len_min < maydf.end_at.values[0] - maydf.begin_at.values[0] < len_max:
            os.remove(mayrem)
    return


def remove_rooms_to_uniform(dir: str, r_to_be: int, rem_dict: {}):
    import os
    roomfiles = fs_utils.get_dir_filepaths(dir)
    if len(roomfiles) <= 3 * r_to_be:
        raise Exception("The classes seem to be uniform, already")

    for i in range(len(roomfiles)):
        mayrem = roomfiles[i]
        maydfcls = p.read_pickle(mayrem).class_name.values[0]
        if maydfcls in rem_dict.keys() and rem_dict[maydfcls] > 0:
            os.remove(mayrem)
            rem_dict[maydfcls] -= 1
    return


def get_len_max_of_signals_in_dir(dir: str) -> int:
    maxL = -1
    signals = fs_utils.get_dir_filepaths(dir)
    for i in range(len(signals)):
        sig = p.read_pickle(signals[i]).output.values[0]

        l = len(sig)
        if l > maxL:
            maxL = l
    return maxL


def get_len_max_of_rirs_in_dir(dir: str) -> int:
    maxL = -1
    rirfile = fs_utils.get_dir_filepaths(dir)
    for i in range(len(rirfile)):
        rir = p.read_pickle(rirfile[i]).rir.values[0]
        linrirs = [micsrcdim for micdim in rir for micsrcdim in micdim]

        l = max([len(rir) for rir in linrirs])
        if l > maxL:
            maxL = l
    return maxL
