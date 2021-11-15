from framework.model.sample.shape.shape_sample import ShapeSample
import numpy as np


class RectangleShapeSample(ShapeSample):
    def __init__(self, x: float, y: float):
        # TODO: setter positivity constraint
        self.__x = float(x)
        self.__y = float(y)
        return

    @property
    def x(self) -> float:
        return self.__x

    @property
    def y(self) -> float:
        return self.__y

    @property
    def is_squared(self) -> bool:
        return self.x == self.y

    # [a][b], a="0-based #corner counter-cw", b="0-based coord (x,y)"
    @property
    def corners(self) -> "np.array":
        return np.array([[0.0, 0.0],
                         [self.x, 0.0],
                         [self.x, self.y],
                         [0.0, self.y]], dtype=np.float)

    @property
    def surface(self) -> float:
        return self.x * self.y

    @property
    def perimeter(self) -> float:
        return 2 * (self.x + self.y)

    @classmethod
    def from_corners(cls, corners: "np.array"):
        return cls(corners[2][0], corners[2][1])

    def grid(self, step: float, inner_margin: float = 0) -> "np.array":
        xmat, xmask, ymat, ymask = ShapeSample._grid_basic_masks(self.x, self.y, step, inner_margin)
        return ShapeSample._grid_apply_masks(xmat, xmask, ymat, ymask)
