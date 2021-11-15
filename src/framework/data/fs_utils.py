from typing import List
from pathlib import Path
import os


def dir_exists_or_create(dir: str):
    if not Path(dir).is_dir():
        # makes base/!report
        os.makedirs(dir, mode=0o777)
    return


def get_dir_filenames(dir: str) -> List[str]:
    if not Path(dir).is_dir():
        raise NotADirectoryError("Directory doesn't exist!")
    ret = []
    for input_file_name in sorted(os.listdir(dir)):
        # solo se il file è tale ed ha estensione
        if Path("{}/{}".format(dir, input_file_name)).is_file() and '.' in input_file_name:
            ret.append(input_file_name)
    return ret


def get_dir_filepaths(dir: str) -> List[str]:   # SECONDARY
    if not Path(dir).is_dir():
        raise NotADirectoryError("Directory doesn't exist!")
    ret = []
    for input_file_name in get_dir_filenames(dir):
        ret.append("{}/{}".format(dir, input_file_name))
    return ret


def get_dir_filenames_no_ext(dir: str) -> List[str]:
    return [filename.split('.')[0] for filename in get_dir_filenames(dir)]


def get_dir_filenames_unique_part(dir: str) -> List[str]:
    ret = []
    tmp = get_dir_filenames(dir)
    if '_' not in tmp[0]:
        ret = tmp
    else:
        for filename in get_dir_filenames(dir):
            all = filename.split('_')
            ret.append("{}_{}".format('_'.join(all[0:len(all)-1]), all[-1].split('.')[0]))
    return ret


# # DEPRECATED AND INEFFICIENT
# def get_filepath_from_unique_part(input_dir: str, part: str) -> str:    # ENTRY
#     for filepath in get_dir_filepaths(input_dir):
#         if fnmatch.fnmatch(filepath, "*/{}.*".format(part)):
#             return filepath
#     return ""


def get_upper(dirorfilepath: str) -> str:
    return '/'.join(dirorfilepath.split('/')[0:-1])


def get_upper_dir(dir: str):
    if not Path(dir).is_dir():
        raise Exception("dir is not a directory")
    return get_upper(dir)


def get_filepath_dir(filepath: str) -> str:
    if not Path(filepath).is_file():
        raise Exception("filepath is not a file")
    return get_upper(filepath)


def get_filepath_filename(filepath: str) -> str:
    if not Path(filepath).is_file():
        raise Exception("filepath is not a file")
    return filepath.split('/')[-1]
