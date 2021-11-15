import os
from pathlib import Path
from framework.data import fs_utils
import pandas as p
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


def create_sample_image(signals):
    fig, axs = plt.subplots(3, 1, constrained_layout=True)

    for i in range(len(axs)):
        axs[i].set_xlabel('n')
        axs[i].set_ylabel(signals[i]["f_name"])
        axs[i].plot(range(len(signals[i]["vector"])), signals[i]["vector"], linewidth=0.2)
        if signals[i]["xlim"]:
            axs[i].set_xlim(*signals[i]["xlim"])
        if signals[i]["ylim"]:
            axs[i].set_ylim(*signals[i]["ylim"])

    return fig, axs


def create_sample_images(files):
    from framework.experimental.simulation import room_utilities

    ret = []
    for file in files:
        acquisition_df = p.read_pickle(file)
        acquisition = acquisition_df.output.values[0]
        h_df = p.read_pickle(acquisition_df.rir_file_path.values[0])

        # # DBG PURPOSES
        # setup_df = p.read_pickle(h_df.input_file_path.values[0])
        # rasd = room_utilities.praroom_from_setup(setup_df)
        # rasd.plot_rir([(0, 0)])
        # plt.xlim(0, 0.1)
        # plt.ylim(-0.5, 0.5)
        # plt.show()
        # plt.clf()

        h = h_df.rir.values[0][0][0]
        fs, incipit = wavfile.read(acquisition_df.input_file_path.values[0])

        fig, _ = create_sample_image([{"f_name": r"$s_i[n]$", "vector": incipit,
                                       "xlim": None, "ylim": [-0.5, 0.5]},
                                      {"f_name": r"$h_{s_i, m_j}[n]$", "vector": h,
                                       "xlim": [0, 2000], "ylim": None},
                                      {"f_name": r"$m_j[n]$", "vector": acquisition,
                                       "xlim": [4500, 8500], "ylim": [-0.5, 0.5]}])
        ret.append(fig)
    return ret


def save_sample_images(images, report_dir):
    for i in range(len(images)):
        images[i].savefig("{}/sample_{}.png".format(report_dir, i), dpi=300)
        plt.close(images[i])
    return


def get_rnd(files, n: int):
    files2 = np.array(files)
    return np.random.choice(files2, n)


def create_report(dir: str):
    files = fs_utils.get_dir_filepaths(dir)
    report_dir = "./datasets/rooms/setups/noise_acquisitions/!report"

    if not Path(report_dir).is_dir():
        os.makedirs(report_dir, mode=0o777)

    images = create_sample_images(get_rnd(files, 20))
    save_sample_images(images, report_dir)
    return
