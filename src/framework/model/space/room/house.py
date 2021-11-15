from framework.model.space.room.room_space import RoomSpace
from framework.model.sample.room.house import HouseRoomSample
from framework.model.space.shape.house import HouseShapeSpace
import framework.extension.math as me
import numpy as np


class HouseRoomSpace(RoomSpace[HouseShapeSpace, HouseRoomSample]):
    def __init__(self, flat_space: HouseShapeSpace, h_min: float, h_max: float):
        super().__init__(flat_space, h_min, h_max)
        return

    def sample(self, step: float) -> HouseRoomSample:
        flat_sample = self.flat_space.sample(step)
        h = me.draw_point(self.h_min, self.h_max, step, 1)[0]
        return HouseRoomSample(flat_sample, h)

    def grid(self, step: float) -> [HouseRoomSample]:
        ret = []
        rg = np.arange(self.h_min, self.h_max + step, step)
        for flat_sample in self.flat_space.grid(step):
            for h in rg:
                ret.append(HouseRoomSample(flat_sample, h))
        return ret
