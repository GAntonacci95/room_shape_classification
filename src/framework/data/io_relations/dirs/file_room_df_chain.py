import pandas as p


def get_room_df_by_chaining(initial_filepath: str):
    got_needed = False
    current_filepath = initial_filepath
    needed_data = None
    i = 0
    while not got_needed:
        current_data = p.read_pickle(current_filepath)

        if "rir_file_path" in current_data.columns.values:
            current_filepath = current_data.rir_file_path.values[0]
        elif "class_name" in current_data.columns.values:
            got_needed = True
            needed_data = current_data
        else:
            tmp = current_data.input_file_path.values[0]
            if isinstance(tmp, str):
                current_filepath = tmp
            elif isinstance(tmp, list):
                current_filepath = tmp[0]   # chaining from one of the features...
            else:
                raise Exception("get_room_df_by_chaining: something went wrong")
    return needed_data
