from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from framework.model.sample.room.wallable import WallableSample
from framework.model.sample.griddable_sample import GriddableSample
from framework.model.sample.samplable_sample import SamplableSample
from framework.model.sample.shape.house import HouseShapeSample
from framework.model.sample.shape.rectangle import RectangleShapeSample
from framework.model.sample.shape.l import LShapeSample
import numpy as np
import framework.extension.math as me
# import gc


# T non potrebbe essere semplicemente ShapeSample?
T = TypeVar('T', RectangleShapeSample, LShapeSample, HouseShapeSample)


# TODO: IN REALTA' LA GRIGLIA SAREBBE PROPRIETA' CACHED DELLA CLASSE: ricalcolarla è follia
class RoomSample(Generic[T], WallableSample, GriddableSample, SamplableSample, ABC):
    @abstractmethod
    def __init__(self, flat_sample: T, h: float):
        # TODO: setter positivity constraint
        self.__flat_sample = flat_sample
        self.__h = float(h)
        return

    @property
    def flat_sample(self) -> T:
        return self.__flat_sample

    @property
    def h(self) -> float:
        return self.__h

    @property
    def volume(self) -> float:
        return self.flat_sample.surface * self.h

    @property
    def surface(self) -> float:
        return sum([w.area() for w in self.walls_pra])

    @property
    def room_sizing(self) -> str:
        if self.volume <= 40:
            return "small"
        elif 40 < self.volume <= 80:
            return "medium"
        elif 80 < self.volume <= 140:
            return "large"
        else:
            return "extra-large"

    # [i][a][b], i="0-based #parete counter-cw", a="0-based #corner counter-cw", b="0-based coord (x,y,z)"
    # TODO: must add floor and ceiling
    @property
    def walls_corners(self) -> "np.array":
        ret = []
        tmp = self.flat_sample.corners
        floor = []
        ceiling = []
        for i in tmp:
            floor.append([*i, 0.0])
            ceiling.append([*i, self.h])

        ret.append(floor)
        for i in range(0, len(tmp)):
            j = (i+1) if i < len(tmp)-1 else 0
            ret.append([
                [*tmp[i], 0.0],
                [*tmp[j], 0.0],
                [*tmp[j], self.h],
                [*tmp[i], self.h]
            ])
        ret.append(ceiling)
        return np.array(ret)

    def sample(self, step: float, inner_margin: float = 0) -> "np.array":   # 3D point within the shape
        tmp = self.grid(step, inner_margin)
        np.random.seed()
        return tmp[np.random.randint(0, len(tmp))]

    # def sample_nu(self, step: float, n: int, inner_margin: float = 0) -> ["np.array"]:   # 3D point within the shape
    #     tmp = self.grid(step, inner_margin)
    #     return tmp[np.random.choice(range(len(tmp)), n, replace=False)].tolist()

    def __eq__(self, other):
        o: RoomSample = other
        return self.__class__ == other.__class__ and np.array_equal(self.walls_corners, o.walls_corners)

    @staticmethod
    def _grid_basic(w: float, h: float, d: float, step: float) -> "np.array":
        zmat, ymat, xmat = \
            np.mgrid[0:me.step_index(d, step) + 1, 0:me.step_index(h, step) + 1, 0:me.step_index(w, step) + 1] * step
        return xmat, ymat, zmat

    @staticmethod
    def _grid_basic_masks(w: float, h: float, d: float, step: float, inner_margin: float = 0) -> "np.array":
        xmat, ymat, zmat = RoomSample._grid_basic(w, h, d, step)
        xmask = (xmat >= inner_margin) & (xmat <= w - inner_margin)
        ymask = (ymat >= inner_margin) & (ymat <= h - inner_margin)
        zmask = (zmat >= inner_margin) & (zmat <= d - inner_margin)
        return xmat, xmask, ymat, ymask, zmat, zmask

    @staticmethod
    def _grid_apply_masks(xmat, xmask, ymat, ymask, zmat, zmask) -> "np.array":
        return np.array([
            xmat[xmask & ymask & zmask],
            ymat[xmask & ymask & zmask],
            zmat[xmask & ymask & zmask]
        ]).T
