import pandas as p
from typing import List, Dict
from pyroomacoustics import Room, Material
import framework.reflection as refl
import framework.data.io_relations.files.parallelization as pdf
import framework.experimental.simulation.room_utilities as ru
import framework.extension.math as me
from functools import partial
from framework.model.sample.room.room_sample import RoomSample
import numpy as np
from datetime import datetime
import threading


# import gc


# TODO: migliorare la stima dei tempi e fare un refactor di sto monolite...
def __sub_simulate(
        rooms_subset: p.DataFrame,
        materials_descr: List[Dict],
        num_rnd_mic_pos: int,
        f_s: int,
        max_ord: int
) -> p.DataFrame:
    ret = []
    first = True
    beginning = elapsed = None
    prod = int(rooms_subset.shape[0] * len(materials_descr) * num_rnd_mic_pos)
    # hasher = hashlib.sha3_256()
    # encoding = "utf-8"
    wnoise = np.random.randn(int(f_s * 0.3))

    for room_tpl in rooms_subset.itertuples():
        georoom: RoomSample = refl.load_class(room_tpl.module_name, room_tpl.class_name)\
            .from_walls_corners(room_tpl.walls_corners)
        spk_poss = georoom.draw_n(0.1, num_rnd_mic_pos, 1)

        for md in materials_descr:
            m = Material(md)
            for spk_pos in spk_poss:
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
                mic_pos = me.rndv_wrt_ref(spk_pos, 0.02, 0.05)

                oid = {
                    "room_id": room_tpl.Index,
                    "material_descr": md,
                    "spk_pos": spk_pos,
                    "mic_pos": mic_pos,
                    "rir|spk,mic": None,
                    "acquisition": None
                }
                hasher.update(repr(oid).encode(encoding))
                filename = "{}.{}".format(hasher.hexdigest(), "wav")
                output_filepath = "{}/{}".format(output_dir, filename)
                try:
                    ru.simulate(praroom, mic_pos, f_s, spk_pos, wnoise, output_filepath)
                except:
                    print("Debug break on me! WTH went wrong?")

                oid["file_name"] = filename
                ret.append(oid)
                if first:
                    elapsed = (datetime.now() - beginning).total_seconds()
                    print("Thread({}) - {}[\"]x({}[rooms] x {}[mats] x {}[acqs] = {}) -> Forecast({}['])".format(
                        threading.current_thread().ident,
                        elapsed,
                        rooms_subset.shape[0],
                        len(materials_descr),
                        num_rnd_mic_pos,
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


def parallel_simulate(rooms: p.DataFrame, materials_descr: List[Dict], num_rnd_mic_pos: int,
             f_s: int, max_ord: int):
    return pdf.compute_parallel(
        rooms,
        partial(__sub_simulate, materials_descr=materials_descr,
                num_rnd_mic_pos=num_rnd_mic_pos, f_s=f_s, max_ord=max_ord)
    )


def simulate(rooms: p.DataFrame, materials_descr: List[Dict], num_rnd_mic_pos: int,
             f_s: int, max_ord: int) -> p.DataFrame:
    return __sub_simulate(rooms, materials_descr, num_rnd_mic_pos, f_s, max_ord)
