from abc import ABC
from enum import Enum


class BeamCombinationSchemeEnum(Enum):
    DOUBLE_BRACEWELL_4 = 'double-bracewell-4'
    KERNEL_3 = 'kernel-3'
    KERNEL_4 = 'kernel-4'
    KERNEL_5 = 'kernel-5'

class BeamCombinationScheme(ABC):
    pass

class DoubleBracewell4(BeamCombinationScheme):
    pass

class Kernel3(BeamCombinationScheme):
    pass

class Kernel4(BeamCombinationScheme):
    pass

class Kernel5(BeamCombinationScheme):
    pass