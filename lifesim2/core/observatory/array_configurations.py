from abc import ABC, abstractmethod
from enum import Enum

import astropy.units
import numpy as np
from astropy import units as u

from lifesim2.util.matrix import get_2d_rotation_matrix


class ArrayConfigurationEnum(Enum):
    """Enum representing the different array configuration types.
    """
    EMMA_X_CIRCULAR_ROTATION = 'emma-x-circular-rotation'
    EMMA_X_DOUBLE_STRETCH = 'emma-x-double-stretch'
    EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION = 'equilateral-triangle-circular-rotation'
    REGULAR_PENTAGON_CIRCULAR_ROTATION = 'regular-pentagon-circular-rotation'


class ArrayConfiguration(ABC):
    """Class representation of a collector array configuration.
    """

    def __init__(self,
                 baseline_minimum: astropy.units.Quantity,
                 baseline_maximum: astropy.units.Quantity,
                 baseline_ratio: astropy.units.Quantity,
                 modulation_period: astropy.units.Quantity):
        """Constructor method.
        :param baseline_minimum: Minimum value of the baseline in meters
        :param baseline_maximum: Maximum value of the baseline in meters
        :param baseline_ratio: Ratio between the respective baselines
        :param modulation_period: Period for one full array modulation (e.g. rotation period for circular rotation)
        """
        self.baseline_minimum = baseline_minimum
        self.baseline_maximum = baseline_maximum
        self.baseline_ratio = baseline_ratio
        self.modulation_period = modulation_period

        super().__init__()

    @abstractmethod
    def get_collector_positions(self, time: astropy.units.Quantity) -> np.ndarray:
        """Return an array containing the time-dependent x- and y-coordinates of the collectors.
        :param time: Time variable in seconds
        :return: An array containing the coordinates.
        """
        pass


class EmmaXCircularRotation(ArrayConfiguration):
    """Class representation of the Emma-X array configuration with circular rotation of the array.
    """

    def get_collector_positions(self, time: astropy.units.Quantity) -> np.ndarray:
        rotation_matrix = get_2d_rotation_matrix(time, self.modulation_period)
        emma_x_static = self.baseline_minimum / 2 * np.array(
            [[self.baseline_ratio, self.baseline_ratio, -self.baseline_ratio, -self.baseline_ratio], [1, -1, -1, 1]])
        return np.matmul(rotation_matrix, emma_x_static)


class EmmaXDoubleStretch(ArrayConfiguration):
    """Class representation of the Emma-X array configuration with double stretching of the array.
    """

    def get_collector_positions(self, time: float) -> np.ndarray:
        emma_x_static = self.baseline_minimum / 2 * np.array(
            [[self.baseline_ratio, self.baseline_ratio, -self.baseline_ratio, -self.baseline_ratio], [1, -1, -1, 1]])
        return emma_x_static * (1 + self.baseline_maximum / self.baseline_minimum * np.sin(
            2 * np.pi * u.rad / self.modulation_period * time))


class EquilateralTriangleCircularRotation(ArrayConfiguration):
    """Class representation of an equilateral triangle configuration with circular rotation of the array.
    """

    def get_collector_positions(self, time: float) -> np.ndarray:
        units = self.baseline_minimum.unit
        height = np.sqrt(3) / 2 * self.baseline_minimum
        height_to_center = height / 3
        rotation_matrix = get_2d_rotation_matrix(time, self.modulation_period)

        equilateral_triangle_static = np.array(
            [[0, self.baseline_minimum.value / 2, -self.baseline_minimum.value / 2],
             [height.value - height_to_center.value, -height_to_center.value, -height_to_center.value]]) * units

        return np.matmul(rotation_matrix, equilateral_triangle_static)


class RegularPentagonCircularRotation(ArrayConfiguration):
    """Class representation of a regular pentagon configuration with circular rotation of the array.
    """

    def get_collector_positions(self, time: float) -> np.ndarray:
        pass
