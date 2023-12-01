from collections import namedtuple
from enum import Enum

Coordinates = namedtuple('Coordinates', 'x y')


class FITSDataType(Enum):
    SyntheticMeasurement = 0
    Template = 1
