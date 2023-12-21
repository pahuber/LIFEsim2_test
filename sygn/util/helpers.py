from collections import namedtuple
from enum import Enum

Coordinates = namedtuple('Coordinates', 'x y')


class FITSReadWriteType(Enum):
    SyntheticMeasurement = 0
    Template = 1


class DataType(Enum):
    """Enum representing the different data types.
    """
    PLANETARY_SYSTEM_CONFIGURATION = 1
    SPECTRUM_DATA = 2
    SPECTRUM_CONTEXT = 3
    POPULATION_CATALOG = 4
