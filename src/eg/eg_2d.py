from framework.model.sample.shape.l import LShapeSample
from framework.model.sample.shape.rectangle import RectangleShapeSample
from framework.model.space.shape.l import LShapeSpace
from framework.model.space.shape.rectangle import RectangleShapeSpace

import framework.reflection as refl
import matplotlib.pyplot as plt

from typing import List
import pandas as p
from framework.data.hashing import Hasher

from framework.data.io_relations.dirs import one_to_one_or_many


def gen_eg_2d():
    # Questa via non è in grado di generare tutte le possibilità
    data = []
    rspace = RectangleShapeSpace(3, 10)
    lspace = LShapeSpace(5, 10, 3)

    rexs: List[RectangleShapeSample] = rspace.draw_all(1)
    lexs: List[LShapeSample] = lspace.draw_all(1)

    for rex in rexs:
        data.append({
            "module_name": rex.__class__.__module__,
            "class_name": rex.__class__.__name__,
            "corners": rex.corners
        })
    for lex in lexs:
        data.append({
            "module_name": lex.__class__.__module__,
            "class_name": lex.__class__.__name__,
            "corners": lex.corners
        })
    return p.DataFrame(data)


def room_shapes(df: p.DataFrame, params: dict) -> None:
    data = gen_eg_2d()
    myh = Hasher()
    for datum in data.itertuples():
        p.DataFrame([datum]).to_pickle("{}/{}.pkl".format(df.output_dir.values[0], myh.compute_hash(datum)))
    return


def compute(params: dict) -> None:
    one_to_one_or_many.compute_sequential(
        "",
        {"handler": room_shapes, "params": params},
        "./datasets"
    )
    return


def load(params: dict) -> str:
    return one_to_one_or_many.get_output_dir(
        "",
        {"handler": room_shapes, "params": params},
        "./datasets"
    )


# TODO: IL SEGUENTE SARA' DA SPOSTARE PER FARE REPORT
def test_refl_cast_grid_rnd_2d(df: p.DataFrame):
    r = df[df["class_name"] == "RectangleShapeSample"].head(2)
    l = df[df["class_name"] == "LShapeSample"].head(1)

    rex: RectangleShapeSample = refl.load_class(module_name=r["module_name"].values[1], class_name=r["class_name"].values[1])\
        .from_corners(r["corners"].values[1])
    # TODO: rex.sample wrong return type - anche 3D
    lex: LShapeSample = refl.load_class(module_name=l["module_name"].values[0], class_name=l["class_name"].values[0])\
        .from_corners(l["corners"].values[0])
    rgrid = rex.grid(step=1)
    rsaa = rex.sample(step=0.1, inner_margin=1)

    lgrid = lex.grid(step=1)
    lsaa = lex.sample(step=0.1, inner_margin=1)
    plt.figure()
    plt.plot(rgrid[:, 0], rgrid[:, 1], 'o')
    plt.plot(rsaa[0], rsaa[1], 'x')
    plt.show()
    plt.figure()
    plt.plot(lgrid[:, 0], lgrid[:, 1], 'o')
    plt.plot(lsaa[0], lsaa[1], 'x')
    plt.show()
    return
