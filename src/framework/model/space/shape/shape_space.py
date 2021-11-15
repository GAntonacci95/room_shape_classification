from abc import ABC
from typing import TypeVar

from framework.model.sample.shape.house import HouseShapeSample
from framework.model.space.griddable_space import GriddableSpace
from framework.model.space.samplable_space import SamplableSpace
from framework.model.sample.shape.rectangle import RectangleShapeSample
from framework.model.sample.shape.l import LShapeSample

T = TypeVar('T', RectangleShapeSample, LShapeSample, HouseShapeSample)


class ShapeSpace(SamplableSpace[T], GriddableSpace[T], ABC):
    pass
