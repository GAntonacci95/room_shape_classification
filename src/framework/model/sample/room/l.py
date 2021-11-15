from framework.model.sample.room.room_sample import RoomSample
from framework.model.space.shape.l import LShapeSample
import numpy as np


class LRoomSample(RoomSample[LShapeSample]):
    def __init__(self, flat_sample: LShapeSample, h: float):
        super().__init__(flat_sample, h)
        return

    @classmethod
    def from_walls_corners(cls, walls_corners: "np.array"):
        d = walls_corners[-1][3]
        return cls(LShapeSample(walls_corners[0][1][0], walls_corners[5][0][1], d[0], d[1]), d[2])

    def grid(self, step: float, inner_margin: float = 0) -> "np.array":
        xmat, xmask, ymat, ymask, zmat, zmask = RoomSample._grid_basic_masks(
            self.flat_sample.x, self.flat_sample.y, self.h, step, inner_margin)
        xyor = self.flat_sample.xyor(xmat, ymat, inner_margin)
        xmask = xyor & xmask
        ymask = xyor & ymask
        return RoomSample._grid_apply_masks(xmat, xmask, ymat, ymask, zmat, zmask)
