from abc import ABC, abstractmethod
from typing import List
import numpy as np
import sys


# TODO: cercare una strategia per replicare ancor meno il codice wrt SamplableSpace
class SamplableSample(ABC):
    @abstractmethod
    def sample(self, step: float, inner_margin: float = 0) -> "np.array":
        pass

    # @abstractmethod
    # def sample_nu(self, step: float, n: int, inner_margin: float = 0) -> List["np.array"]:
    #     pass

    # finite time implementations
    # TODO: in realtà all è finite time, questa deve esplodere se non riesce a trovarne n
    # TODO: inoltre sarebbe meglio un bel p.DataFrame.sample sulla tabella "intero spazio di combinazioni di variabili"?
    def draw_n(self, step: float, n: int, inner_margin: float = 0) -> List["np.array"]:
        ret = []
        nop = 0
        i = 0
        chk = False
        while nop < 50 and i < n:
            # self: (*ShapeSample | *RoomSample).sample(step, inner_margin) -> "np.array"
            # "np.array" doesn't implement "Equatable"
            o = self.sample(step, inner_margin)
            if not np.any([np.array_equal(o, i) for i in ret]):
                ret.append(o)
                nop = 0
                i += 1
            else:
                nop += 1
                if not chk:
                    # print("Some repetition avoided: the check works!")
                    chk = True
        return ret

    # def draw_n_nu(self, step: float, n: int, inner_margin: float = 0) -> List["np.array"]:
    #     return self.sample_nu(step, n, inner_margin)

    def draw_all(self, step: float, inner_margin: float = 0) -> List["np.array"]:
        return self.draw_n(step, sys.maxsize, inner_margin)
