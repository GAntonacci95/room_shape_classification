from abc import ABC, abstractmethod
import numpy as np


# MAYBE WallableSample is a CornerableSample


class CornerableSample(#WallableBaseSample[Wall2D],
                        ABC):
    # representation that shall be saved within a DataFrame
    @property
    @abstractmethod
    def corners(self) -> "np.array":
        pass

    @classmethod
    @abstractmethod
    def from_corners(cls, corners: "np.array"):     # constructor given the representation above (from DataFrame item)
        pass

    # representation accepted by Room.from_corners classmethod
    @property
    def corners_pra(self) -> "np.array":
        return self.corners.T

    # @property
    # def walls(self) -> List[Wall2D]:
    #     ret = []
    #     for i in range(0, len(self.corners)):
    #         j = (i+1) if i < len(self.corners)-1 else 0
    #         ret.append(Wall2D([self.corners[i], self.corners[j]]))
    #     return ret
