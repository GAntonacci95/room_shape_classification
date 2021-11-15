# CURRENT

import numpy as np
import pandas as p
import framework.reflection as refl
from framework.model.sample.room.room_sample import RoomSample
from pyroomacoustics import Room, Material
from pyroomacoustics import constants as praconsts


def georoom_from_room_df(room_frame: p.DataFrame) -> RoomSample:
    return refl.load_class(room_frame.module_name.values[0], room_frame.class_name.values[0]) \
        .from_walls_corners(room_frame.walls_corners.values[0])


def georoom_from_setup_df(setup_frame: p.DataFrame) -> RoomSample:
    room_of_setup = p.read_pickle(setup_frame.input_file_path.values[0])
    return georoom_from_room_df(room_of_setup)


def ignorant_v_to_t60_map(v_range: dict, t60_range: dict, v: float) -> float:
    v_min, v_max = v_range["min"], v_range["max"]
    t60_min, t60_max = t60_range["min"], t60_range["max"]
    return (v - v_min) / (v_max - v_min) * (t60_max - t60_min) + t60_min


# selfish method of RoomSample?
def e_absorption(room: RoomSample, t60: float, c: float = praconsts.get('c')) -> float:
    # sole 3D case
    sab_coef = 24
    return sab_coef * np.log(10) * room.volume / (c * t60 * room.surface)


def praroom_from_setup_df(setup_frame: p.DataFrame,
                          material_arg: any, f_s: int, max_ord: int, use_ray: bool) -> Room:
    georoom = georoom_from_setup_df(setup_frame)
    m = Material(material_arg)

    # here, instead of extruding, georoom.walls_pra might be used as well
    praroom = Room.from_corners(corners=georoom.flat_sample.corners_pra,
                                fs=f_s,
                                max_order=max_ord,
                                ray_tracing=use_ray,
                                materials=m,
                                air_absorption=True)
    praroom.extrude(height=georoom.h, materials=m)

    # from all sources
    for pos_src in setup_frame.pos_srcs.values[0]:
        praroom.add_source(np.array(pos_src).reshape((3, 1)))   # (3,) -> (3, 1)
    # to one mic
    praroom.add_microphone(np.array(setup_frame.pos_mic.values[0]).reshape((3, 1)))  # (3,) -> (3, 1)
    return praroom


def praroom_set_srcs_signal(praroom: Room, signals: ["np.array"]) -> None:
    if praroom.n_sources != len(signals):
        raise Exception("Acqus: error, n_srcs and len(sigs) mismatch")
    for i in range(len(praroom.sources)):
        praroom.sources[i].add_signal(signals[i])
    return


# # CHECK ALTERNATIVE...
# def praroom_from_setup(setup_frame: p.DataFrame, signal = None) -> Room:
#     georoom = georoom_from_setup(setup_frame)
#     m = Material(setup_frame.material_descriptor.values[0])
#
#     praroom = Room.from_corners(corners=georoom.flat_sample.corners_pra,
#                                 fs=setup_frame.f_s.values[0],
#                                 max_order=setup_frame.max_ord.values[0],
#                                 materials=m,
#                                 air_absorption=True)
#     praroom.extrude(height=georoom.h, materials=m)
#
#     # from 1 source
#     praroom.add_source(setup_frame.pos_src.values[0], signal)
#     # to 1 (near field) + n_rnd_acquisitons_per_room microphones
#     for mic_pos in setup_frame.pos_mics.values[0]:
#         praroom.add_microphone(mic_pos, setup_frame.f_s.values[0])
#     return praroom
