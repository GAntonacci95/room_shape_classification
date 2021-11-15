from pathlib import Path
import os

import pandas

from eg.features.post_processing import eg_finalize_ds
from eg.reports.eg_reports_feat_maps import get_rnd_ids, create_sample_images_2, save_sample_images

# mado di quante schifezze si sta riempiendo tutto quanto...
def create_report(dsname: str, feature_names: [str], win_len: int, hop_size: int, folder_spec: str = ""):
    import numpy as np
    df: pandas.DataFrame = eg_finalize_ds.load(dsname)
    report_dir = "./datasets/nn/features/post_processing/final_ds_feat_maps_normal/!report_{}".format(folder_spec)

    if not Path(report_dir).is_dir():
        os.makedirs(report_dir, mode=0o777)
    ids = get_rnd_ids(len(df), 20)
    images = create_sample_images_2(df.iloc[ids].X.values, feature_names, win_len, hop_size)
    save_sample_images(images, report_dir)
    return
