from framework.model.space.shape.shape_space import ShapeSpace
from framework.model.sample.shape.l import LShapeSample
import framework.extension.math as me
import numpy as np


class LShapeSpace(ShapeSpace[LShapeSample]):
    def __init__(self, side_min: float, side_max: float, side_min_min: float):
        self.__side_min = float(side_min)
        self.__side_max = float(side_max)
        self.__side_min_min = float(side_min_min)
        return

    @property
    def side_min(self) -> float:
        return self.__side_min

    @property
    def side_max(self) -> float:
        return self.__side_max

    @property
    def side_min_min(self) -> float:
        return self.__side_min_min

    def sample(self, step: float) -> LShapeSample:
        if self.side_min_min > me.first_ge_step_value(self.side_min / 2, step):
            raise Exception("Wrong init")
        x, y = me.draw_point(self.side_min, self.side_max, step, 2)
        x2, y2 = me.draw_point(me.first_le_step_value(self.side_min_min, step),
            me.first_ge_step_value(np.min([x, y]) / 2, step), step, 2)
        return LShapeSample(x, y, x2, y2)

    def grid(self, step: float) -> [LShapeSample]:
        ret = []
        rg = np.arange(self.side_min, self.side_max + step, step)
        for x in rg:
            for y in rg:
                rg2 = np.arange(
                    me.first_le_step_value(self.side_min_min, step),
                    me.first_ge_step_value(np.min([x, y]) / 2, step) + step,
                    step)
                for x2 in rg2:
                    for y2 in rg2:
                        ret.append(LShapeSample(x, y, x2, y2))
        return ret
