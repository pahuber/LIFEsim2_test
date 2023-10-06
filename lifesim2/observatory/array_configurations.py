from abc import ABC
from enum import Enum

import numpy as np


class ArrayConfigurationEnum(Enum):
    EMMA_X_CIRCULAR_ROTATION = 'emma-x-circular-rotation'
    EMMA_X_DOUBLE_STRETCH = 'emma-x-double-stretch'
    EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION = 'equilateral-triangle-circular-rotation'
    REGULAR_PENTAGON_CIRCULAR_ROTATION = 'regular-pentagon-circular-rotation'


class ArrayConfiguration(ABC):
    def __init__(self, type: str):
        pass


class EmmaXCircularRotation(ArrayConfiguration):
    def __init__(self):
        pass

class EmmaXDoubleStretch(ArrayConfiguration):
    def __init__(self):
        pass

class EquilateralTriangleCircularRotation(ArrayConfiguration):
    def __init__(self):
        pass

class RegularPentagonCircularRotation(ArrayConfiguration):
    def __init__(self):
        pass
