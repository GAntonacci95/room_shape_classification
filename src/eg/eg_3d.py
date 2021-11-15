from framework.model.sample.room.house import HouseRoomSample
from framework.model.sample.room.l import LRoomSample
from framework.model.sample.room.rectangle import RectangleRoomSample
from framework.model.space.room.house import HouseRoomSpace
from framework.model.space.room.l import LRoomSpace
from framework.model.space.room.rectangle import RectangleRoomSpace
from framework.model.space.shape.house import HouseShapeSpace
from framework.model.space.shape.l import LShapeSpace
from framework.model.space.shape.rectangle import RectangleShapeSpace

from typing import List
import pandas as p
from framework.data.hashing import Hasher

from framework.data.io_relations.dirs import one_to_one_or_many


def gen_eg_3d():
    # Questa via non è in grado di generare tutte le possibilità
    data = []
    rspace = RectangleRoomSpace(RectangleShapeSpace(3, 10), 3, 7)
    lspace = LRoomSpace(LShapeSpace(5, 10, 3), 3, 7)

    rexs: List[RectangleRoomSample] = rspace.draw_all(1)
    lexs: List[LRoomSample] = lspace.draw_all(1)

    for rex in rexs:
        data.append({
            "module_name": rex.__class__.__module__,
            "class_name": rex.__class__.__name__,
            "walls_corners": rex.walls_corners
        })
    for lex in lexs:
        data.append({
            "module_name": lex.__class__.__module__,
            "class_name": lex.__class__.__name__,
            "walls_corners": lex.walls_corners
        })
    return p.DataFrame(data)


def rooms(df: p.DataFrame, params: dict) -> None:
    data = gen_eg_3d()
    myh = Hasher()
    for datum in data.itertuples():
        p.DataFrame([datum]).to_pickle("{}/{}.pkl".format(df.output_dir.values[0], myh.compute_hash(datum)))
    return


def compute(params: dict) -> None:
    # GENERATION HERE MUST BE KEPT SEQUENTIAL for rooms uniqueness
    one_to_one_or_many.compute_sequential(
        "",
        {"handler": rooms, "params": params},
        "./datasets"
    )
    return


def load(params: dict) -> str:
    return one_to_one_or_many.get_output_dir(
        "",
        {"handler": rooms, "params": params},
        "./datasets"
    )


def obj_samps(params: dict):
    # V \in [27, 2250]m3
    rspace = RectangleRoomSpace(RectangleShapeSpace(params["R"]["side_min"], params["R"]["side_max"]),
                                params["R"]["h_min"], params["R"]["h_max"])
    # V \in [81, 1070]m3
    lspace = LRoomSpace(LShapeSpace(params["L"]["side_min"], params["L"]["side_max"], params["L"]["side_min_min"]),
                                params["L"]["h_min"], params["L"]["h_max"])
    # V \in [33.75, 1250]m3
    #                                          10 per costruzione: qui y non è h: h=y+y_roof
    hspace = HouseRoomSpace(HouseShapeSpace(params["H"]["side_min"], params["H"]["side_max"]),
                                params["H"]["h_min"], params["H"]["h_max"])

    # 63stanze/classe * 20bande = 1260stanze/classe
    rexs: List[RectangleRoomSample] = rspace.grid_volume_band_uniform(
        params["R"]["room_vars_step"],
        params["R"]["v_min"],
        params["R"]["v_max"],
        params["R"]["v_bandwidth"],
        params["R"]["n_rooms_per_band"]
    )
    lexs: List[LRoomSample] = lspace.grid_volume_band_uniform(
        params["L"]["room_vars_step"],
        params["L"]["v_min"],
        params["L"]["v_max"],
        params["L"]["v_bandwidth"],
        params["L"]["n_rooms_per_band"]
    )
    hexs: List[HouseRoomSample] = hspace.grid_volume_band_uniform(
        params["H"]["room_vars_step"],
        params["H"]["v_min"],
        params["H"]["v_max"],
        params["H"]["v_bandwidth"],
        params["H"]["n_rooms_per_band"]
    )
    return rexs, lexs, hexs


def gen_eg_3d_rlh(params: dict):
    # Questa via non è in grado di generare tutte le possibilità
    data = []
    rexs, lexs, hexs = obj_samps(params)

    for ex in [*rexs, *lexs, *hexs]:
        data.append({
            "module_name": ex.__class__.__module__,
            "class_name": ex.__class__.__name__,
            "walls_corners": ex.walls_corners
        })
    return p.DataFrame(data)


def rooms_rlh(df: p.DataFrame, params: dict) -> None:
    data = gen_eg_3d_rlh(params)
    myh = Hasher()
    for datum in data.itertuples():
        p.DataFrame([datum]).to_pickle("{}/{}.pkl".format(df.output_dir.values[0], myh.compute_hash(datum)))
    return


def compute_rlh(params: dict) -> None:
    # GENERATION HERE MUST BE KEPT SEQUENTIAL for rooms uniqueness
    one_to_one_or_many.compute_sequential(
        "",
        {"handler": rooms_rlh, "params": params},
        "./datasets"
    )
    return


def load_rlh(params: dict) -> str:
    return one_to_one_or_many.get_output_dir(
        "",
        {"handler": rooms_rlh, "params": params},
        "./datasets"
    )
