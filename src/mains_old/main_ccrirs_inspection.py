import pickle as pkl
import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt


def load_all(path11, path21, path12, path22):
    with open(path11, "rb") as i11:
        rirs1 = pkl.load(i11)
    with open(path21, "rb") as i21:
        rirs2 = pkl.load(i21)
    with open(path12, "rb") as i12:
        vols1 = pkl.load(i12)
    with open(path22, "rb") as i22:
        vols2 = pkl.load(i22)
    return rirs1, rirs2, vols1, vols2


def get_l_max(rirs):
    rir = None
    Lmax = -1
    ptr = (-1, -1, -1)
    for a in range(len(rirs)):
        for b in range(len(rirs[a])):
            for c in range(len(rirs[a][b])):
                tmprir = rirs[a][b][c]
                if len(tmprir) > Lmax:
                    ptr = (a, b, c)
                    rir = tmprir
                    Lmax = len(tmprir)
    return ptr, rir, Lmax


# main application
def run():
    path11, path21 = "/nas/home/ccastelnuovo/Tesi/larger_my_parallel_dataset.pickle", \
                   "/nas/home/ccastelnuovo/Tesi/new_larger_my_parallel_dataset.pickle"
    path12, path22 = "/nas/home/ccastelnuovo/Tesi/larger_my_parallel_target.pickle", \
                   "/nas/home/ccastelnuovo/Tesi/new_larger_my_parallel_target.pickle"

    rirs1, rirs2, vols1, vols2 = load_all(path11, path21, path12, path22)

    info1, info2 = get_l_max(rirs1), get_l_max(rirs2)
    v1, v2 = vols1[info1[0][0]], vols2[info2[0][0]]
    # CONSIDERAZIONE:
    # nel suo ds ci sono rir che durano ~4s
    # tuttavia lui ha anche volumi di ordine 5...
    # non mi torna che le rir più lunghe non siano associate per forza alle stanze con volumi più grandi...

    # OTHER TEST
    # geo rooms
    fs = int(16E3)
    ccdim = [15, 25, 4]
    absorption = {
        "east": 0.2,
        "west": 0.2,
        "north": 0.2,
        "south": 0.2,
        "ceiling": 0.2,
        "floor": 0.2
    }
    m = pra.Material(0.2)
    max_order = 10  # 50 LOL!
    # first arg ok from https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.room.html
    ccstyleroom = pra.ShoeBox(ccdim, fs=fs, absorption=absorption, max_order=max_order)

    corners = np.array([[0, 15, 15, 0], [0, 0, 25, 25]])
    mystyleroom = pra.Room.from_corners(corners, fs=fs, max_order=max_order, materials=m,
                                        air_absorption=True)
    mystyleroom.extrude(height=ccdim[-1], materials=m)

    testrooms = [ccstyleroom, mystyleroom]

    # mic and src
    mic, src = np.array([[1], [1], [1]]), np.array([[3], [3], [3]])     # np.array([[14], [24], [3]])
    for room in testrooms:
        room.add_microphone(loc=mic, fs=fs)
        room.add_source(position=src)
        room.compute_rir()
        room.plot_rir([(0, 0)])
        plt.show()
    # CONSIDERAZIONE:
    # in effetti dalla stampa sembrano durare ~0.7s~11Ksamp,
    # ma non vedo differenze tra quanto ho fatto nel codice ed in questo test:
    # il vincolo sulla distanza mic-src influisce sulla lunghezza della rir
    return
