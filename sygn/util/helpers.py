from collections import namedtuple
from enum import Enum

Coordinates = namedtuple('Coordinates', 'x y')


class FITSDataType(Enum):
    SyntheticData = 0
    Templates = 1
