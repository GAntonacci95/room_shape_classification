import pandas as p


def get_pre_feat_df_by_chaining(initial_filepath: str):
    got_needed = False
    current_filepath = initial_filepath
    needed_data = None
    i = 0
    while not got_needed:
        current_data = p.read_pickle(current_filepath)

        # eg adapt or fdrsCC or noise_acquisitions or voice_acquisitions
        if "adapt" in current_filepath or "fdrs" in current_filepath or "acquisition" in current_filepath:
            got_needed = True
            needed_data = current_data
        else:
            tmp = current_data.input_file_path.values[0]
            if isinstance(tmp, str):
                current_filepath = tmp
            elif isinstance(tmp, list):
                current_filepath = tmp[0]   # chaining from one of the features...
            else:
                raise Exception("get_pre_feat_df_by_chaining: something went wrong")
    return needed_data
