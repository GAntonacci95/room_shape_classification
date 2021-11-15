from framework.model.sample.room.room_sample import RoomSample
from framework.model.space.shape.house import HouseShapeSample
import numpy as np


class HouseRoomSample(RoomSample[HouseShapeSample]):
    def __init__(self, flat_sample: HouseShapeSample, h: float):
        super().__init__(flat_sample, h)
        return

    @classmethod
    def from_walls_corners(cls, walls_corners: "np.array"):
        x, y, h = walls_corners[-1][2]
        ry = walls_corners[-1][3][1] - y
        return cls(HouseShapeSample(x, y, ry), h)

    def grid(self, step: float, inner_margin: float = 0) -> "np.array":
        xmat, xmask, ymat, ymask, zmat, zmask = RoomSample._grid_basic_masks(
            self.flat_sample.x, self.flat_sample.y + self.flat_sample.roof_y, self.h, step, inner_margin)
        margin_q = inner_margin / np.cos(self.flat_sample.roof_angle)
        ymask = self.flat_sample.ymask12(xmat, ymat, margin_q) & ymask
        return RoomSample._grid_apply_masks(xmat, xmask, ymat, ymask, zmat, zmask)
