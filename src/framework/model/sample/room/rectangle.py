from framework.model.sample.room.room_sample import RoomSample
from framework.model.sample.shape.rectangle import RectangleShapeSample
import numpy as np


class RectangleRoomSample(RoomSample[RectangleShapeSample]):
    def __init__(self, flat_sample: RectangleShapeSample, h: float):
        super().__init__(flat_sample, h)
        return

    @classmethod
    def from_walls_corners(cls, walls_corners: "np.array"):
        d = walls_corners[-1][2]
        return cls(RectangleShapeSample(d[0], d[1]), d[2])

    def grid(self, step: float, inner_margin: float = 0) -> "np.array":
        xmat, xmask, ymat, ymask, zmat, zmask = RoomSample._grid_basic_masks(
            self.flat_sample.x, self.flat_sample.y, self.h, step, inner_margin)
        return RoomSample._grid_apply_masks(xmat, xmask, ymat, ymask, zmat, zmask)
