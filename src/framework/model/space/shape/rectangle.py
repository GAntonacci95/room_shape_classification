from framework.model.space.shape.shape_space import ShapeSpace
from framework.model.sample.shape.rectangle import RectangleShapeSample
import numpy as np
import framework.extension.math as me


class RectangleShapeSpace(ShapeSpace[RectangleShapeSample]):
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

    def sample(self, step: float) -> RectangleShapeSample:
        x, y = me.draw_point(self.side_min, self.side_max, step, 2)
        return RectangleShapeSample(x, y)

    @staticmethod
    def _grid_basic(w: float, h: float, step: float) -> "np.array":
        ymat, xmat = np.mgrid[0:me.step_index(h, step) + 1, 0:me.step_index(w, step) + 1] * step
        return xmat, ymat

    def grid(self, step: float) -> [RectangleShapeSample]:
        ret = []
        rg = np.arange(self.side_min, self.side_max + step, step)
        for x in rg:
            for y in rg:
                ret.append(RectangleShapeSample(x, y))
        return ret
