from abc import ABC, abstractmethod
from typing import List
from pyroomacoustics import Wall
import numpy as np


class WallableSample(ABC):
    # representation that shall be saved within a DataFrame
    @property
    @abstractmethod
    def walls_corners(self) -> "np.array":
        pass

    # constructor given the representation above (from DataFrame item)
    @classmethod
    @abstractmethod
    def from_walls_corners(cls, walls_corners: "np.array"):
        pass

    # representation accepted by Room.from_corners classmethod
    @property
    def walls_pra(self) -> List[Wall]:
        ret = []
        for wall_corners in self.walls_corners:
            ret.append(Wall(np.array(wall_corners).T))
        return ret
