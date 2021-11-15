import os
from pathlib import Path
from framework.data import fs_utils
import pandas as p
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


def create_sample_image(signal, f_s: float = 16000):
    fig = plt.figure()
    plt.xlabel(signal["f_dom"])
    plt.ylabel(signal["f_name"])
    tax = np.linspace(0, len(signal["vector"]) / f_s, len(signal["vector"]))
    plt.plot(tax, signal["vector"], linewidth=0.2)
    if signal["xlim"]:
        plt.xlim(*signal["xlim"])
    if signal["ylim"]:
        plt.xlim(*signal["ylim"])
    plt.tight_layout()

    return fig


def create_sample_images(files):
    ret = []
    for file in files:
        h = p.read_pickle(file).rir.values[0][0][0]
        fig = create_sample_image({"f_dom": "t [s]", "f_name": r"$h_{s_i, m_j}(t)$", "vector": h,
                                       "xlim": None, "ylim": None})
        ret.append(fig)
    return ret


def create_sample_audios(files):
    ret = []
    for file in files:
        h = p.read_pickle(file).rir.values[0][0][0]
        ret.append(h)
    return ret


def save_sample_images(images, report_dir, images_names):
    for i in range(len(images)):
        images[i].savefig("{}/{}.png".format(report_dir, images_names[i]), dpi=300)
        plt.close(images[i])
    return


def save_sample_audios(audios, report_dir, audios_names):
    from scipy.io import wavfile
    for i in range(len(audios)):
        # both floating and int16 samples supported
        wavfile.write("{}/{}.wav".format(report_dir, audios_names[i]), 16000, audios[i])
    return


def get_rnd(files, n: int):
    files2 = np.array(files)
    return np.random.choice(files2, n)


def get_RLH_sizing_samples(files):
    from framework.experimental.simulation import room_utilities as ru
    classes = ["RectangleRoomSample", "LRoomSample", "HouseRoomSample"]
    sizings = ["third1", "third2", "third3"]
    needed = {c: {t: 1 for t in sizings} for c in classes}
    ret = []
    for rirfile in files:
        if any([needed[c][t] for c in classes for t in sizings]):
            rir_df = p.read_pickle(rirfile)
            setup_df = p.read_pickle(rir_df.input_file_path.values[0])
            room_df = p.read_pickle(setup_df.input_file_path.values[0])
            georoom = ru.georoom_from_room_df(room_df)
            cla, sz = georoom.__class__.__name__, "third2"

            if georoom.volume < 100:
                sz = "third1"
            elif georoom.volume > 666:
                sz = "third3"

            if needed[cla][sz] == 1:
                ret.append({"file_path": rirfile, "class_label": cla, "sizing": sz})
                needed[cla][sz] = 0
        else:
            break
    return ret


def dist_from_counts(unique, counts):
    return unique, counts / counts.sum()


def len_freq(rirfiles):
    lens = []
    for i in range(len(rirfiles)):
        tmp = p.read_pickle(rirfiles[i]).rir.values[0]
        for ms in range(len(tmp)):
            for sm in range(len(tmp[ms])):
                lens.append((len(tmp[ms][sm]) // 100) * 100)

    unique, counts = np.unique(lens, return_counts=True)
    return unique, counts


def create_len_freq_image(unique, counts, report_dir):
    fig = plt.figure()
    plt.bar(unique, counts, width=1000)
    plt.xlabel(r"$len$")
    plt.ylabel(r"$COUNT_{len}$")
    plt.savefig("{}/lens_distro.png".format(report_dir))
    plt.close(fig)
    return


def t60_freq(rirfiles, rounding_decimals: int):
    t60s = []
    for i in range(len(rirfiles)):
        tmp = p.read_pickle(rirfiles[i]).t60.values[0]
        for ms in range(len(tmp)):
            for sm in range(len(tmp[ms])):
                t60s.append(np.round(tmp[ms][sm], rounding_decimals))

    unique, counts = np.unique(t60s, return_counts=True)
    return unique, counts


def create_t60_freq_image(unique, counts, report_dir):
    fig = plt.figure()
    plt.bar(unique, counts, width=0.05)
    plt.xlabel(r"$T_{60}$")
    plt.ylabel(r"$COUNT_{T_{60}}$")
    plt.savefig("{}/t60s_distro.png".format(report_dir))
    plt.close(fig)
    return


def create_report(dir: str):
    rirfiles = fs_utils.get_dir_filepaths(dir)
    report_dir = "{}/!report".format(fs_utils.get_upper_dir(dir))

    if not Path(report_dir).is_dir():
        os.makedirs(report_dir, mode=0o777)

    toreport = get_RLH_sizing_samples(rirfiles)  # get_rnd(rirfiles, 2)     # TODO 20 if not WC
    input_files = [d["file_path"] for d in toreport]
    output_fnames = ["sample_{}_{}".format(d["class_label"], d["sizing"]) for d in toreport]

    images = create_sample_images(input_files)
    save_sample_images(images, report_dir, output_fnames)
    audios = create_sample_audios(input_files)
    save_sample_audios(audios, report_dir, output_fnames)
    lunique, lcounts = len_freq(rirfiles)
    create_len_freq_image(lunique, lcounts, report_dir)
    tunique, tcounts = t60_freq(rirfiles, 2)
    create_t60_freq_image(tunique, tcounts, report_dir)
    return
