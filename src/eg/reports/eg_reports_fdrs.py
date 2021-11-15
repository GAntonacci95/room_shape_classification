import os
from pathlib import Path
from framework.data import fs_utils
import pandas as p
import numpy as np
import matplotlib.pyplot as plt


def create_sample_images(files, report_dir):
    fig = plt.figure()
    for i in range(len(files)):
        fdrfiles = files[i]
        dfs = [p.read_pickle(fdrfile) for fdrfile in fdrfiles]
        signal = p.read_pickle(dfs[0].input_file_path.values[0]).output.values[0]
        signal_fdrs = [(df.begin_at.values[0], df.end_at.values[0]) for df in dfs]

        plt.plot(range(len(signal)), signal)
        for begin_at, end_at in signal_fdrs:
            plt.axvspan(begin_at, end_at, color="red", alpha=0.3)
        plt.xlabel(r"$N_{sample}$")
        plt.ylabel('y')
        plt.savefig("{}/sample_{}.png".format(report_dir, i))
        plt.clf()
    plt.close(fig)
    return


def len_dist(unique, counts):
    return unique, counts / counts.sum()


def len_freq(fdrfiles):
    lens = []
    for i in range(len(fdrfiles)):
        tmp = p.read_pickle(fdrfiles[i])
        begin_at, end_at = tmp.begin_at.values[0], tmp.end_at.values[0]
        lens.append(end_at - begin_at)
    unique, counts = np.unique(lens, return_counts=True)
    return unique, counts


def create_len_freq_image(unique, counts, report_dir):
    fig = plt.figure()
    plt.bar(unique, counts)
    plt.xlabel = r"$len$"
    plt.ylabel = r"$COUNT_{len}$"
    plt.savefig("{}/lens_distro.png".format(report_dir))
    plt.close(fig)
    return


def get_groups_by_common_input(fdrfiles):
    return list(set(['_'.join(fdrfile.split('_')[0:-1]) for fdrfile in fdrfiles]))


def get_rnd_groups_by_common_input(fdrfiles, n: int):
    groups = np.array(get_groups_by_common_input(fdrfiles))
    indices = np.random.randint(0, len(groups), n)
    return [[fdrfile for fdrfile in fdrfiles if group in fdrfile] for group in groups[indices]]


def create_report(fdrs_dir: str, t: str):
    fdrfiles = fs_utils.get_dir_filepaths(fdrs_dir)
    report_dir = "./datasets/nn/features/pre_processing/{}/!report".format(t)

    if not Path(report_dir).is_dir():
        os.makedirs(report_dir, mode=0o777)

    create_sample_images(get_rnd_groups_by_common_input(fdrfiles, 100), report_dir)
    unique, counts = len_freq(fdrfiles)
    create_len_freq_image(unique, counts, report_dir)
    return
