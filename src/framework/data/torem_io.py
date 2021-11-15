import pandas as p
from pathlib import Path
import shutil
import os


def renew_dir(directory: str):
    if Path(directory).is_dir():
        shutil.rmtree(directory)
    os.mkdir(directory, mode=0o755)
    return


def monolite_to_split(base_dir: str, df_file_name: str,
                      input_file_path_field_name: str, output_field: str):
    monolite = "{}/{}".format(base_dir, df_file_name)

    input_dir = "{}/input".format(base_dir)
    output_dir = "{}/output".format(base_dir)
    io_assoc = "{}/io_assoc.pkl".format(base_dir)



    df: p.DataFrame = p.read_pickle(monolite)
    renew_dir(input_dir)
    renew_dir(output_dir)
    for t in df.itertuples():
        input_file_name = t[input_file_path_field_name].split("/")[-1]
        output_file_path = "{}/{}".format(output_dir, input_file_name)
        p.DataFrame(t[output_field]).to_pickle(output_file_path)



    os.remove(monolite)
    return


def run_for_file_in_input(base_dir: str, handler, handler_params: dict):
    input_dir = "{}/input".format(base_dir)
    output_dir = "{}/output".format(base_dir)
    io_assoc = "{}/io_assoc.json".format(base_dir)
    if not Path(input_dir).is_dir():
        raise Exception("{} doesn't exist!".format(input_dir))
    renew_dir(output_dir)

    for _, _, input_file_names in os.walk(input_dir):
        # filter? [file_name for file_name in input_file_names if ".wav" in file_name]
        for input_file_name in input_file_names:
            input_file_path = "{}/{}".format(input_dir, input_file_name)

    return
