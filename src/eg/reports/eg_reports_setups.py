import os
from pathlib import Path
from framework.data import fs_utils
import pandas as p

from framework.experimental.simulation import room_utilities as ru
import matplotlib.pyplot as plt


# TODO: SOLO QUESTO METODO VA POTENZIALMENTE CAMBIATO
#  QUANDO CAMBIERO' LA STRUTTURA DEL setup_df
def create_sample_image(setup_df, room_df):
    from eg.reports.eg_reports_rooms import create_sample_image as csri
    import numpy as np
    fig, ax = csri(room_df)

    ex = ru.georoom_from_room_df(room_df)
    grid = ex.flat_sample.grid(0.1, 1)  # first arg <= min{room_generation_step}

    asrcs = np.array(setup_df.pos_srcs.values[0])
    amic = setup_df.pos_mic.values[0]
    mat = setup_df.material_descriptor.values[0]

    fig.text(0.6, 0.03, r"$z_{s_i} = " + "{:.2f}".format(asrcs[0, 2]) +
                 " [m], z_{m_j} = " + "{:.2f}".format(amic[2]) +
             " [m]$", fontsize=7)
    fig.text(0.6, 0.01, "mat_coeffs = " + str(mat["coeffs"] if isinstance(mat, dict) else mat), fontsize=7)

    plt.plot(grid[:, 0], grid[:, 1], 'o', color="green")
    plt.plot(asrcs[:, 0], asrcs[:, 1], 'o', color="red", label=r"$s_i$")
    plt.plot(amic[0], amic[1], 'x', color="red", label=r"$m_j$")
    ax.legend()

    return fig, ax


def create_sample_images(setup_files, classes, n_per_class: int):
    images_info = {c: [] for c in classes}

    for c, lst in zip(images_info.keys(), images_info.values()):
        for setup_file in setup_files:
            setup_df = p.read_pickle(setup_file)
            room_df = p.read_pickle(setup_df.input_file_path.values[0])
            k = room_df.class_name.values[0]
            if k == c:
                if len(lst) < n_per_class:
                    f, _ = create_sample_image(setup_df, room_df)
                    lst.append(f)
                else:   # break inner cycle
                    break
    return images_info


def save_sample_images(images_info, report_dir: str):
    for c, lst in zip(images_info.keys(), images_info.values()):
        for i in range(len(lst)):
            lst[i].savefig("{}/{}_{}.png".format(report_dir, c, i), dpi=300)
            plt.close(lst[i])
    return


def create_report(setups_dir: str, classes: []):
    setup_files = fs_utils.get_dir_filepaths(setups_dir)
    report_dir = "{}/!report".format(fs_utils.get_upper_dir(setups_dir))

    if not Path(report_dir).is_dir():
        os.makedirs(report_dir, mode=0o777)

    images_info = create_sample_images(setup_files, classes, 1)     # TODO 2 if not WC
    save_sample_images(images_info, report_dir)
    return
