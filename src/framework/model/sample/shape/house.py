from framework.model.sample.shape.shape_sample import ShapeSample
import numpy as np


class HouseShapeSample(ShapeSample):
    def __init__(self, x: float, y: float, roof_y: float):
        # TODO: setter positivity constraint
        self.__x = float(x)
        self.__y = float(y)
        self.__roof_y = float(roof_y)
        return

    @property
    def x(self) -> float:
        return self.__x

    @property
    def y(self) -> float:
        return self.__y

    @property
    def roof_y(self) -> float:
        return self.__roof_y

    @property
    def roof_slope(self) -> float:
        return self.roof_y / (self.x / 2)

    @property
    def roof_angle(self) -> float:
        return np.arctan(self.roof_slope)

    # [a][b], a="0-based #corner counter-cw", b="0-based coord (x,y)"
    @property
    def corners(self) -> "np.array":
        return np.array([[0.0, 0.0],
                         [self.x, 0.0],
                         [self.x, self.y],
                         [self.x / 2, self.y + self.roof_y],
                         [0.0, self.y]], dtype=np.float)

    @property
    def surface(self) -> float:
        return (self.x * self.y) + (self.x * self.roof_y / 2)

    @property
    def perimeter(self) -> float:
        from framework.extension.math import distance
        return self.x + 2 * (self.y + distance(self.corners[2], self.corners[3]))

    @classmethod
    def from_corners(cls, corners: "np.array"):
        return cls(
            corners[1][0],
            corners[2][1],
            corners[3][1] - corners[2][1]
        )

    def __f1(self, dom):
        return self.roof_slope * dom + self.y

    def __f2(self, dom):
        return -self.roof_slope * (dom - self.x) + self.y

    def __ymask1(self, xmat, ymat, margin_q: float):
        f1 = self.__f1(xmat) - margin_q
        return ymat <= f1

    def __ymask2(self, xmat, ymat, margin_q: float):
        f2 = self.__f2(xmat) - margin_q
        return ymat <= f2

    def ymask12(self, xmat, ymat, margin_q):
        return self.__ymask1(xmat, ymat, margin_q) & self.__ymask2(xmat, ymat, margin_q)

    def grid(self, step: float, inner_margin: float = 0) -> "np.array":
        xmat, xmask, ymat, ymask = ShapeSample._grid_basic_masks(self.x, self.y + self.roof_y, step, inner_margin)
        margin_q = inner_margin / np.cos(self.roof_angle)
        ymask = self.ymask12(xmat, ymat, margin_q) & ymask
        return ShapeSample._grid_apply_masks(xmat, xmask, ymat, ymask)
