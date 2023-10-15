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
        self.number_of_transmission_maps = len(self.get_differential_intensity_response_indices())

    @abstractmethod
    def get_beam_combination_transfer_matrix(self) -> np.ndarray:
        """Return the bea, combination transfer matrix.
        :return: An array representing the bea combination transfer matrix
        """
        pass

    @abstractmethod
    def get_differential_intensity_response_indices(self) -> list:
        """Return the pairs of indices of the intensity response vector that make up a transmission map.

        :return: List of tuples containing the pairs of indices
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

    def get_differential_intensity_response_indices(self) -> list:
        return [(2, 3)]


class Kernel3(BeamCombinationScheme):
    """Class representation of a Kernel nulling beam combination scheme.
    """

    def get_beam_combination_transfer_matrix(self) -> np.ndarray:
        return 1 / np.sqrt(3) * np.array([[1, 1, 1],
                                          [1, np.exp(2j * np.pi / 3), np.exp(4j * np.pi / 3)],
                                          [1, np.exp(4j * np.pi / 3), np.exp(2j * np.pi / 3)]])

    def get_differential_intensity_response_indices(self) -> list:
        return [(1, 2)]


class Kernel4(BeamCombinationScheme):
    """Class representation of a Kernel nulling beam combination scheme.
    """
    pass


class Kernel5(BeamCombinationScheme):
    """Class representation of a Kernel nulling beam combination scheme.
    """
    pass
