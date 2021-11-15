from abc import ABC, abstractmethod
from framework.model.sample.shape.cornerable import CornerableSample
from framework.model.sample.griddable_sample import GriddableSample
from framework.model.sample.samplable_sample import SamplableSample
import numpy as np
import framework.extension.math as me
# import gc


class ShapeSample(CornerableSample, GriddableSample, SamplableSample, ABC):
    def sample(self, step: float, inner_margin: float = 0) -> "np.array":   # 2D point within the shape
        tmp = self.grid(step, inner_margin)
        np.random.seed()
        return tmp[np.random.randint(0, len(tmp))]

    # def sample_nu(self, step: float, n: int, inner_margin: float = 0) -> ["np.array"]:   # 3D point within the shape
    #     tmp = self.grid(step, inner_margin)
    #     return tmp[np.random.choice(range(len(tmp)), n, replace=False)].tolist()

    @property
    @abstractmethod
    def surface(self) -> float:
        pass

    @property
    @abstractmethod
    def perimeter(self) -> float:
        pass

    @staticmethod
    def _grid_basic(w: float, h: float, step: float) -> "np.array":
        ymat, xmat = np.mgrid[0:me.step_index(h, step) + 1, 0:me.step_index(w, step) + 1] * step
        return xmat, ymat

    @staticmethod
    def _grid_basic_masks(w: float, h: float, step: float, inner_margin: float = 0) -> "np.array":
        xmat, ymat = ShapeSample._grid_basic(w, h, step)
        xmask = (xmat >= inner_margin) & (xmat <= w - inner_margin)
        ymask = (ymat >= inner_margin) & (ymat <= h - inner_margin)
        return xmat, xmask, ymat, ymask

    @staticmethod
    def _grid_apply_masks(xmat, xmask, ymat, ymask) -> "np.array":
        return np.array([xmat[xmask & ymask], ymat[xmask & ymask]]).T

    def __eq__(self, other):
        o: ShapeSample = other
        return self.__class__ == other.__class__ and np.array_equal(self.corners, o.corners)
