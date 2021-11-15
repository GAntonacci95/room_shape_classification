import os
from pathlib import Path

import librosa

from framework.data import fs_utils
import pandas as p
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


def create_basic(signal):
    fig = plt.figure()
    plt.xlabel = 'n'
    plt.ylabel = r"$m_j[n]$"
    plt.plot(range(len(signal)), signal, linewidth=0.2)
    return fig


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


def create_melspectrum_image(signal):
    from librosa.feature import melspectrogram
    from librosa.display import specshow
    # S (n_melbins, n_tbins)
    S = melspectrogram(y=signal, sr=16000)
    d = librosa.power_to_db(S, ref=np.max)
    fig = plt.figure(figsize=(10, 4))
    specshow(data=d, x_axis="time", sr=16000, y_axis="mel", fmax=8000)
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    return fig


def create_spectrum_image(signal):
    from scipy.signal import spectrogram
    # Zxx (n_fbins, n_tbins)
    fbins, tbins, Sxx = spectrogram(x=signal, fs=16000, nperseg=1024, noverlap=512, nfft=2048)
    fig = plt.figure()
    plt.pcolormesh(tbins, fbins, Sxx, shading='gouraud')
    plt.xlabel('t [s]')
    plt.ylabel('f [Hz]')
    plt.tight_layout()
    return fig


def create_sample_images(files):
    import framework.extension.math as me

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

        h = h_df.rir.values[0][0][0]    # considero la rir tra solo mic e prima sorgente
        fs, incipit = wavfile.read(acquisition_df.input_file_path.values[0][0])     # ed il segnale sorgente associato
        incipit = me.normalize_signal(incipit, -3)

        fig, _ = create_sample_image([{"f_name": r"$s_i[n]$", "vector": incipit,
                                       "xlim": None, "ylim": [-0.8, 0.8]},
                                      {"f_name": r"$h_{s_i, m_j}[n]$", "vector": h,
                                       "xlim": None, "ylim": None},
                                      {"f_name": r"$m_j[n]$", "vector": acquisition,
                                       "xlim": None, "ylim": [-0.8, 0.8]}])
        ret.append(fig)
    return ret


def create_melspectrum_images(files):
    ret = []
    for file in files:
        acq_df = p.read_pickle(file)
        ret.append(create_melspectrum_image(acq_df.output.values[0]))
    return ret


def create_spectrum_images(files):
    ret = []
    for file in files:
        acq_df = p.read_pickle(file)
        ret.append(create_spectrum_image(acq_df.output.values[0]))
    return ret


def create_sample_audios(files):
    ret = []
    for file in files:
        sig = p.read_pickle(file).output.values[0]
        ret.append(sig)
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
    for acqfile in files:
        if any([needed[c][t] for c in classes for t in sizings]):
            acq_df = p.read_pickle(acqfile)
            rir_df = p.read_pickle(acq_df.rir_file_path.values[0])
            setup_df = p.read_pickle(rir_df.input_file_path.values[0])
            room_df = p.read_pickle(setup_df.input_file_path.values[0])
            georoom = ru.georoom_from_room_df(room_df)
            cla, sz = georoom.__class__.__name__, "third2"

            if georoom.volume < 100:
                sz = "third1"
            elif georoom.volume > 666:
                sz = "third3"

            if needed[cla][sz] == 1:
                ret.append({"file_path": acqfile, "class_label": cla, "sizing": sz})
                needed[cla][sz] = 0
        else:
            break
    return ret


def create_report(dir: str, spec: str):
    files = fs_utils.get_dir_filepaths(dir)
    report_dir = "./datasets/rooms/setups/acquisitions/!report/{}".format(spec)

    if not Path(report_dir).is_dir():
        os.makedirs(report_dir, mode=0o777)

    toreport = get_RLH_sizing_samples(files)  # get_rnd(rirfiles, 2)     # TODO 20 if not WC
    input_files = [d["file_path"] for d in toreport]
    output_fnames = ["{}_{}_sample".format(d["class_label"], d["sizing"]) for d in toreport]
    specmel_output_fnames = ["{}_{}_specMel".format(d["class_label"], d["sizing"]) for d in toreport]
    spec_output_fnames = ["{}_{}_spec".format(d["class_label"], d["sizing"]) for d in toreport]

    images = create_sample_images(input_files)
    save_sample_images(images, report_dir, output_fnames)
    specmel_images = create_melspectrum_images(input_files)
    save_sample_images(specmel_images, report_dir, specmel_output_fnames)
    spec_images = create_spectrum_images(input_files)
    save_sample_images(spec_images, report_dir, spec_output_fnames)
    audios = create_sample_audios(input_files)
    save_sample_audios(audios, report_dir, output_fnames)
    return
