import pandas as p
from typing import List, Dict
from pyroomacoustics import Room, Material, MicrophoneArray
import framework.data.io_relations.files.parallelization as pdf
import framework.extension.math as me
from functools import partial
import framework.reflection as refl
from framework.model.sample.room.room_sample import RoomSample
import numpy as np
from datetime import datetime
import threading


# TODO: migliorare la stima dei tempi e fare un refactor di sto monolite...
# i segnali dei microfoni sono signal * irs con pad fino alla lunghezza massima
def __sub_simulate(
        rooms_subset: p.DataFrame,
        materials_descr: List[Dict],
        n_rnd_acquisitons_per_room: int,                 # num of srcs and colocated mics
        f_s: int,
        max_ord: int,
        n_further_rnd_mic_pos: int = 0      # num of non-colocated rnd mics
) -> p.DataFrame:
    ret = []
    first = True
    beginning = elapsed = None
    prod = int(rooms_subset.shape[0] * len(materials_descr) * n_rnd_acquisitons_per_room)

    for room_tpl in rooms_subset.itertuples():
        georoom: RoomSample = refl.load_class(room_tpl.module_name, room_tpl.class_name)\
            .from_walls_corners(room_tpl.walls_corners)
        pos = georoom.draw_n(0.1, n_rnd_acquisitons_per_room + n_further_rnd_mic_pos, 1)
        pos_srcs = pos[0:n_rnd_acquisitons_per_room]    # da 0 ne prendo n
        further_pos_mics = pos[n_rnd_acquisitons_per_room:]
        del pos

        for md in materials_descr:
            m = Material(md)
            for pos_src in pos_srcs:
                if first:
                    beginning = datetime.now()
                    print("Thread({}) - Start({})".format(
                        threading.current_thread().ident,
                        beginning
                    ))

                # è necessario creare una praroom per ogni misurazione nonostante sia meno efficiente...
                # TODO: max_ord dovrebbe essere valutato tramite iSabine
                # l'uso del ray_tracing accresce i tempi computazionali (x30 ca) - schivare come la peste
                praroom = Room.from_corners(corners=georoom.flat_sample.corners_pra,
                                            fs=f_s,
                                            max_order=max_ord,
                                            materials=m,
                                            air_absorption=True)
                praroom.extrude(height=georoom.h, materials=m)
                pos_mics = [me.rndv_wrt_ref(pos_src, 0.02, 0.05)]
                for i in further_pos_mics:
                    pos_mics.append(i)

                # SETUP & COMPUTE_IRS
                try:
                    # from 1 source
                    praroom.add_source(pos_src.T)
                    # to 1 (near field) + n_rnd_acquisitons_per_room microphones
                    praroom.add_microphone_array(MicrophoneArray(np.array(pos_mics).T, f_s))
                    praroom.compute_rir()
                except Exception as e:     # ho rispettato la notazione, se qualcosa non va debug!
                    print("Debug break on me! WTH went wrong?")

                # SAVE_IRS
                # praroom.rir: List[List["np.ndarray"]]
                # praroom.rir[i][j]: ir_ij(k) = rir(k|src_i, mic_j)
                # i segnali con cui convolveremo, vuoi rumori o voci anecoiche hanno lunghezze diverse,
                # perciò effettuare il padding delle risposte risulta inutile
                ret.append({
                    "room_id": room_tpl.Index,
                    "material_descr": md,
                    "pos_src": pos_src,
                    "pos_mics": pos_mics,
                    "irs": praroom.rir
                })

                if first:
                    elapsed = (datetime.now() - beginning).total_seconds()
                    print("Thread({}) - {}[\"]x({}[rooms] x {}[mats] x {}[acqs] = {}) -> Forecast({}['])".format(
                        threading.current_thread().ident,
                        elapsed,
                        rooms_subset.shape[0],
                        len(materials_descr),
                        n_rnd_acquisitons_per_room,
                        prod,
                        int(elapsed * prod / 60)
                    ))
                    first = False

        #         del output_filepath, praroom, m     # not needed for oid
        #         gc.collect()
        # del src_poss, georoom     # not needed for oid
        # gc.collect()

    end = datetime.now()
    elapsed = (end - beginning).total_seconds()
    print("Thread({}) - End({}) - Actual({}['])".format(
        threading.current_thread().ident,
        end,
        int(elapsed / 60)
    ))
    return p.DataFrame(ret)


def parallel_simulate(
        rooms: p.DataFrame,
        materials_descr: List[Dict],
        n_rnd_acquisitons_per_room: int,
        f_s: int,
        max_ord: int,
        n_further_rnd_mic_pos: int = 0
):
    return pdf.compute_parallel(
        rooms,      # parallelize passes a subset to __sub_simulate
        partial(__sub_simulate,
                materials_descr=materials_descr,
                n_rnd_acquisitons_per_room=n_rnd_acquisitons_per_room,
                f_s=f_s,
                max_ord=max_ord,
                n_further_rnd_mic_pos=n_further_rnd_mic_pos
                )
    )


def simulate(
        rooms: p.DataFrame,
        materials_descr: List[Dict],
        n_rnd_acquisitons_per_room: int,
        f_s: int,
        max_ord: int,
        n_further_rnd_mic_pos: int = 0
) -> p.DataFrame:
    return __sub_simulate(
        rooms,
        materials_descr,
        n_rnd_acquisitons_per_room,
        f_s,
        max_ord,
        n_further_rnd_mic_pos
    )
