from typing import List
from framework.data.io_relations.dirs import many_to_one
from framework.data import fs_utils
import pandas as p
from framework.data.thread_timing import ThreadTimer
import numpy as np


def feat_maps(df: p.DataFrame, params: dict) -> None:
    print("FeatMaps generation")
    tt = ThreadTimer()
    input_dirs = df.input_dirs.values[0]    # shared across rows
    dir_filenames = fs_utils.get_dir_filenames(input_dirs[0])   # IS NOT shared across all dirs
    dirs_ls = [fs_utils.get_dir_filepaths(input_dir) for input_dir in input_dirs]

    for t in df.itertuples():
        tt.start()
        i = 0
        # the following cycle is needed to identify t.input_file_name unique part
        # into dir_filenames, for sake of efficiency
        for i in range(len(dir_filenames)):
            if t.input_file_name == dir_filenames[i] or t.input_file_name in dir_filenames[i]:
                break
        feature_files = [dir_ls[i] for dir_ls in dirs_ls]
        names = [fs_utils.get_filepath_filename(dir_ls[i]) for dir_ls in dirs_ls]
        name_match = [t.input_file_name == name or t.input_file_name in name for name in names]
        if not all(name_match):
            raise Exception("FeatMaps: something went wrong while rebuilding the maps")
        tmp = [p.read_pickle(feature).output.values[0] for feature in feature_files]

        mapee = np.concatenate(tmp, axis=0)

        datum = {
            "input_file_path": feature_files,
            "output": mapee
        }
        p.DataFrame([datum]).to_pickle("{}/{}.pkl".format(t.output_dir, t.input_file_name))
        tt.end_first(df.shape[0])
    tt.end_total()
    return


def compute(anypreproc_output_dirs: List[str], params: dict) -> None:
    many_to_one.compute_parallel(
        anypreproc_output_dirs,
        {"handler": feat_maps, "params": params},
        "./datasets/nn/features/post_processing"
    )
    return


def load(anypreproc_output_dirs: List[str], params: dict) -> str:
    return many_to_one.get_output_dir(
        anypreproc_output_dirs,
        {"handler": feat_maps, "params": params},
        "./datasets/nn/features/post_processing"
    )


def load_shape(loaded_dir: str) -> any:
    df = p.read_pickle(fs_utils.get_dir_filepaths(loaded_dir)[0])
    return df.output.values[0].shape
