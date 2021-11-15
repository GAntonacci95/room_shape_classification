from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List
import sys


T = TypeVar('T')


# TODO: cercare una strategia per replicare ancor meno il codice wrt SamplableSample
class SamplableSpace(Generic[T], ABC):
    @abstractmethod
    def sample(self, step: float) -> T:
        pass

    # finite time implementation
    def draw_n(self, step: float, n: int) -> List[T]:
        ret = []
        nop = 0
        i = 0
        # chk = False
        while nop < 50 and i < n:
            # self: (*ShapeSpace | *RoomSpace).sample(step) -> (*ShapeSample | *RoomSample) (T)
            # T implementations must implement "Equatable"
            o = self.sample(step)
            if o not in ret:
                ret.append(o)
                nop = 0
                i += 1
            else:
                nop += 1
                # if not chk:
                #     print("Some repetition avoided: the check works!")
                #     chk = True
        return ret

    def draw_all(self, step: float) -> List[T]:
        return self.draw_n(step, sys.maxsize)
