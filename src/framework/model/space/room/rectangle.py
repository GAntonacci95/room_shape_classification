from framework.model.space.room.room_space import RoomSpace
from framework.model.space.shape.rectangle import RectangleShapeSpace
from framework.model.sample.room.rectangle import RectangleRoomSample
import framework.extension.math as me
import numpy as np


class RectangleRoomSpace(RoomSpace[RectangleShapeSpace, RectangleRoomSample]):
    def __init__(self, flat_space: RectangleShapeSpace, h_min: float, h_max: float):
        super().__init__(flat_space, h_min, h_max)
        return

    def sample(self, step: float) -> RectangleRoomSample:
        flat_sample = self.flat_space.sample(step)
        h = me.draw_point(self.h_min, self.h_max, step, 1)[0]
        return RectangleRoomSample(flat_sample, h)

    def grid(self, step: float) -> [RectangleRoomSample]:
        ret = []
        rg = np.arange(self.h_min, self.h_max + step, step)
        for flat_sample in self.flat_space.grid(step):
            for h in rg:
                ret.append(RectangleRoomSample(flat_sample, h))
        return ret
