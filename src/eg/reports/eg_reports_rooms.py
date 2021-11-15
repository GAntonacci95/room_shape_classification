import os
from pathlib import Path
from typing import List
from framework.data import fs_utils
import pandas as p

from framework.experimental.simulation import room_utilities as ru
import matplotlib.pyplot as plt


def create_sample_image(room_df):
    import numpy as np
    ex = ru.georoom_from_room_df(room_df)
    corners = ex.flat_sample.corners
    corners2 = np.roll(corners, -1, axis=0)

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.text(0, 0.01, r"$z_{room} = " + "{:.2f}".format(ex.h) + " [m]$", fontsize=7)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.plot(corners[:, 0], corners[:, 1], '-o', color="blue")
    plt.plot(corners2[:, 0], corners2[:, 1], '-o', color="blue")

    return fig, ax


def create_sample_images(room_files, classes, n_per_class: int):
    images_info = {c: [] for c in classes}

    for c, lst in zip(images_info.keys(), images_info.values()):
        for room_file in room_files:
            room_df = p.read_pickle(room_file)
            k = room_df.class_name.values[0]
            if k == c:
                if len(lst) < n_per_class:
                    f, _ = create_sample_image(room_df)
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


def create_classes_dist(room_files, classes):
    dist = {c: 0 for c in classes}
    for room_file in room_files:
        room_df = p.read_pickle(room_file)
        dist[room_df.class_name.values[0]] += 1
    return {c: dist[c]/len(room_files) for c in classes}


def create_classes_dist_image(dist):
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    ax.set_ylabel(r"$p_c$")
    plt.bar(range(len(list(dist.keys()))), dist.values(), tick_label=list(dist.keys()))
    return fig


def save_classes_dist_image(dist_image, report_dir):
    dist_image.savefig("{}/classes_distro.png".format(report_dir), dpi=300)
    plt.close(dist_image)
    return


def create_classes_vol_dist(room_files, classes):
    import numpy as np
    dist = {c: {} for c in classes}
    for room_file in room_files:
        room_df = p.read_pickle(room_file)
        ex = ru.georoom_from_room_df(room_df)
        sub_dict = dist[room_df.class_name.values[0]]
        sub_dict_k = str(ex.volume)
        if sub_dict_k not in sub_dict.keys():
            sub_dict[sub_dict_k] = 1
        else:
            sub_dict[sub_dict_k] += 1

    ret = {}
    for c, sub_dict in zip(dist.keys(), dist.values()):
        c_vols = np.array([float(vol) for vol in sub_dict.keys()])
        c_cnts = np.array([int(cnt) for cnt in sub_dict.values()])
        p_vol_given_c = c_cnts / np.sum(c_cnts)
        ret[c] = {"c_vols": c_vols, "p_vol_given_c": p_vol_given_c}
    return ret


def create_classes_vol_dist_image(classes_vol_dist):
    ks, vs = list(classes_vol_dist.keys()), list(classes_vol_dist.values())
    fig, axs = plt.subplots(len(ks), 1, constrained_layout=True)
    fig.suptitle(r"$p(V | c_i)$")
    for r in range(len(ks)):
        c = ks[r]
        c_x, c_y = vs[r]["c_vols"], vs[r]["p_vol_given_c"]
        axs[r].set_xlabel(r"$V [m^3]$")
        axs[r].set_ylabel(r"p(V | c='" + c[0] + "')")
        axs[r].bar(c_x, c_y)

    return fig


def save_classes_vol_dist_image(classes_vol_dist_info, report_dir):
    classes_vol_dist_info.savefig("{}/vol_dist_given_c".format(report_dir), dpi=300)
    return


def create_report(rooms_dir: str, classes: List[str]):
    room_files = fs_utils.get_dir_filepaths(rooms_dir)
    report_dir = "{}/!report".format(fs_utils.get_upper_dir(rooms_dir))

    if not Path(report_dir).is_dir():
        os.makedirs(report_dir, mode=0o777)

    images_info = create_sample_images(room_files, classes, 2)
    save_sample_images(images_info, report_dir)

    classes_dist = create_classes_dist(room_files, classes)
    classes_dist_image = create_classes_dist_image(classes_dist)
    save_classes_dist_image(classes_dist_image, report_dir)

    classes_vol_dist = create_classes_vol_dist(room_files, classes)
    classes_vol_dist_info = create_classes_vol_dist_image(classes_vol_dist)
    save_classes_vol_dist_image(classes_vol_dist_info, report_dir)
    return
