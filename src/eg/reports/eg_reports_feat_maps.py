import os
from pathlib import Path
from framework.data import fs_utils
import pandas as p
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


def inframe(squared, feature_names: [str], win_len: int = None, hop_size: int = None):
    fig = plt.figure()
    if win_len and hop_size:
        plt.xlabel = "n'(N = {}, H = {})".format(win_len, hop_size)
    else:
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
    squaring_factor = int(squared.shape[0] / len(feature_names))
    plt.ylabel = "features"
    plt.yticks(range(int(squaring_factor/2),
                     len(feature_names) * squaring_factor,
                     squaring_factor), feature_names)
    plt.imshow(squared)     # prints the image to fig object
    plt.colorbar()
    plt.tight_layout()
    return fig


def create_sample_image(feature_map, feature_names: [str], win_len: int, hop_size: int):
    squaring_factor = int(feature_map.shape[1] / feature_map.shape[0])
    squared = np.empty([feature_map.shape[0] * squaring_factor, feature_map.shape[1]])
    for r in range(feature_map.shape[0]):
        squared[r * squaring_factor:(r+1) * squaring_factor, :] = \
            np.repeat(feature_map[r, :].reshape([1, -1]), squaring_factor, axis=0)

    return inframe(squared, feature_names, win_len, hop_size)


def create_sample_images_2(fmaps: [], feature_names: [str], win_len: int, hop_size: int):
    ret = []
    for fmap in fmaps:
        fig = create_sample_image(fmap, feature_names, win_len, hop_size)
        ret.append(fig)
    return ret


def create_sample_images(files, feature_names: [str], win_len: int, hop_size: int):
    feature_maps = []
    for file in files:
        feature_maps.append(p.read_pickle(file).output.values[0])
    return create_sample_images_2(feature_maps, feature_names, win_len, hop_size)


def save_sample_images(images, report_dir):
    for i in range(len(images)):
        images[i].savefig("{}/sample_{}.png".format(report_dir, i), dpi=300)
        plt.close(images[i])
    return


def get_rnd_ids(max: int, n: int):
    np.random.seed(666)
    return np.random.randint(0, max, n)


def get_rnd(files, n: int):
    files2 = np.array(files)
    return files2[get_rnd_ids(len(files2), n)]


def create_report(dir: str, feature_names: [str], win_len: int, hop_size: int):
    files = fs_utils.get_dir_filepaths(dir)
    report_dir = "./datasets/nn/features/post_processing/feat_maps/!report"

    if not Path(report_dir).is_dir():
        os.makedirs(report_dir, mode=0o777)

    images = create_sample_images(get_rnd(files, 20), feature_names, win_len, hop_size)
    save_sample_images(images, report_dir)
    return
