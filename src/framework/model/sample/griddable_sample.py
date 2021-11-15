from abc import ABC, abstractmethod
import numpy as np


class GriddableSample(ABC):
    @abstractmethod
    def grid(self, step: float, inner_margin: float = 0) -> "np.array":
        pass
