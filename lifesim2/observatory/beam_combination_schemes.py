from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class BeamCombinationSchemeEnum(Enum):
    """Enum representing the different beam combination schemes.
    """
    DOUBLE_BRACEWELL = 'double-bracewell'
    KERNEL_3 = 'kernel-3'
    KERNEL_4 = 'kernel-4'
    KERNEL_5 = 'kernel-5'


class BeamCombinationScheme(ABC):
    """Class representation of a beam combination scheme.
    """

    def __init__(self):
        """Constructor method.
        """
        super().__init__()
        self.number_of_inputs = self.get_beam_combination_transfer_matrix().shape[1]
        self.number_of_outputs = self.get_beam_combination_transfer_matrix().shape[0]

    @abstractmethod
    def get_beam_combination_transfer_matrix(self) -> np.ndarray:
        """Return the bea, combination transfer matrix.
        :return: An array representing the bea combination transfer matrix
        """
        pass


class DoubleBracewell(BeamCombinationScheme):
    """Class representation of a double Bracewell beam combination scheme.
    """

    def get_beam_combination_transfer_matrix(self) -> np.ndarray:
        return 1 / np.sqrt(4) * np.array([[0, 0, np.sqrt(2), np.sqrt(2)],
                                          [np.sqrt(2), np.sqrt(2), 0, 0],
                                          [1, -1, -np.exp(1j * np.pi / 2), np.exp(1j * np.pi / 2)],
                                          [1, -1, np.exp(1j * np.pi / 2), -np.exp(1j * np.pi / 2)]])


class Kernel3(BeamCombinationScheme):
    """Class representation of a Kernel nulling beam combination scheme.
    """
    pass


class Kernel4(BeamCombinationScheme):
    """Class representation of a Kernel nulling beam combination scheme.
    """
    pass


class Kernel5(BeamCombinationScheme):
    """Class representation of a Kernel nulling beam combination scheme.
    """
    pass
