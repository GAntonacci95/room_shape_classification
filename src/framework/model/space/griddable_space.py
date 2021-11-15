from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List


T = TypeVar('T')


class GriddableSpace(Generic[T], ABC):
    @abstractmethod
    def grid(self, step: float) -> List[T]:
        pass
