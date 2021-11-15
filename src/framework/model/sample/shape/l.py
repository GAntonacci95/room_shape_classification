from framework.model.sample.shape.shape_sample import ShapeSample
import numpy as np


class LShapeSample(ShapeSample):
    def __init__(self, x: float, y: float, x2: float, y2: float):
        # TODO: setter positivity constraint
        self.__x = float(x)
        self.__y = float(y)
        self.__x2 = float(x2)
        self.__y2 = float(y2)
        return

    @property
    def x(self) -> float:
        return self.__x

    @property
    def y(self) -> float:
        return self.__y

    @property
    def x2(self) -> float:
        return self.__x2

    @property
    def y2(self) -> float:
        return self.__y2

    # [a][b], a="0-based #corner counter-cw", b="0-based coord (x,y)"
    @property
    def corners(self) -> "np.array":
        return np.array([[0.0, 0.0],
                         [self.x, 0.0],
                         [self.x, self.y2],
                         [self.x2, self.y2],
                         [self.x2, self.y],
                         [0.0, self.y]], dtype=np.float)

    @property
    def surface(self) -> float:
        return (self.x * self.y) - ((self.x - self.x2) * (self.y - self.y2))

    @property
    def perimeter(self) -> float:
        return 2 * (self.x + self.y)

    @classmethod
    def from_corners(cls, corners: "np.array"):
        return cls(
            corners[1][0],
            corners[5][1],
            corners[3][0],
            corners[2][1]
        )

    def xyor(self, xmat, ymat, inner_margin):
        return (xmat <= self.x2 - inner_margin) | (ymat <= self.y2 - inner_margin)

    def grid(self, step: float, inner_margin: float = 0) -> "np.array":
        xmat, xmask, ymat, ymask = ShapeSample._grid_basic_masks(self.x, self.y, step, inner_margin)
        xyor = self.xyor(xmat, ymat, inner_margin)
        xmask = xyor & xmask
        ymask = xyor & ymask
        return ShapeSample._grid_apply_masks(xmat, xmask, ymat, ymask)
