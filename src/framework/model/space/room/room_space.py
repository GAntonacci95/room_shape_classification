from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from framework.model.sample.room.house import HouseRoomSample
from framework.model.sample.room.l import LRoomSample
from framework.model.sample.room.rectangle import RectangleRoomSample
from framework.model.space.griddable_space import GriddableSpace
from framework.model.space.samplable_space import SamplableSpace
from framework.model.space.shape.house import HouseShapeSpace
from framework.model.space.shape.l import LShapeSpace
from framework.model.space.shape.rectangle import RectangleShapeSpace


T1 = TypeVar('T1', RectangleShapeSpace, LShapeSpace, HouseShapeSpace)
T2 = TypeVar('T2', RectangleRoomSample, LRoomSample, HouseRoomSample)


class RoomSpace(Generic[T1, T2], SamplableSpace[T2], GriddableSpace[T2], ABC):
    @abstractmethod
    def __init__(self, flat_space: T1, h_min: float, h_max: float):
        # TODO: setter positivity constraint
        self.__flat_space = flat_space
        self.__h_min = h_min
        self.__h_max = h_max
        return

    @property
    def flat_space(self) -> T1:
        return self.__flat_space

    @property
    def h_min(self) -> float:
        return self.__h_min

    @property
    def h_max(self) -> float:
        return self.__h_max

    def grid_volume_band_uniform(self, step: float,
                                 v_start: int, v_end: int, v_band_width: int,
                                 n_per_band: int) -> [T2]:
        import numpy as np
        ret = []
        tmp = [{"bandstart": bandstart, "bandend": bandend, "lst": []} for bandstart, bandend in \
            zip(range(v_start, v_end-v_start+1, v_band_width), range(v_start+v_band_width, v_end+1, v_band_width))]

        g = self.grid(step)
        np.random.shuffle(g)
        for samp in g:
            if not all([len(d["lst"]) == n_per_band for d in tmp]):
                for d in tmp:
                    if d["bandstart"] <= samp.volume < d["bandend"] and len(d["lst"]) < n_per_band:
                        d["lst"].append(samp)
                        ret.append(samp)
            else:
                break
        if not all([len(d["lst"]) == n_per_band for d in tmp]):
            raise Exception("Unexpected behavior")
        return ret
