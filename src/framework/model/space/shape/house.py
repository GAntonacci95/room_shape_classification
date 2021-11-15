from framework.model.space.shape.shape_space import ShapeSpace
from framework.model.sample.shape.house import HouseShapeSample
import framework.extension.math as me
import numpy as np


class HouseShapeSpace(ShapeSpace[HouseShapeSample]):
    def __init__(self, side_min: float, side_max: float):
        self.__side_min = float(side_min)
        self.__side_max = float(side_max)
        return

    @property
    def side_min(self) -> float:
        return self.__side_min

    @property
    def side_max(self) -> float:
        return self.__side_max

    def sample(self, step: float) -> HouseShapeSample:
        x, y = me.draw_point(self.side_min, self.side_max, step, 2)
        rfyfrom, rfyto = me.first_le_step_value(np.min([x, y]) / 2, step), \
                         me.first_ge_step_value(np.max([x, y]) / 2, step)
        roof_y = me.draw_point(rfyfrom, rfyto, step, 1)[0]
        return HouseShapeSample(x, y, roof_y)

    def grid(self, step: float) -> [HouseShapeSample]:
        ret = []
        rg = np.arange(self.side_min, self.side_max + step, step)
        for x in rg:
            for y in rg:
                rg2 = np.arange(
                    me.first_le_step_value(np.min([x, y]) / 2, step),
                    me.first_ge_step_value(np.max([x, y]) / 2, step) + step,
                    step)
                for roof_y in rg2:
                    ret.append(HouseShapeSample(x, y, roof_y))
        return ret
