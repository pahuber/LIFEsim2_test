from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import astropy.units
import numpy as np
from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from sygn.core.modules.target_system.star import Star
from sygn.io.validators import validate_quantity_units
from sygn.util.matrix import get_2d_rotation_matrix


class ArrayConfigurationEnum(Enum):
    """Enum representing the different array configuration types.
    """
    EMMA_X_CIRCULAR_ROTATION = 'emma-x-circular-rotation'
    EMMA_X_DOUBLE_STRETCH = 'emma-x-double-stretch'
    EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION = 'equilateral-triangle-circular-rotation'
    REGULAR_PENTAGON_CIRCULAR_ROTATION = 'regular-pentagon-circular-rotation'


class ArrayConfiguration(ABC, BaseModel):
    """Class representation of a collector array configuration.
    """
    baseline_minimum: Any
    baseline_maximum: Any
    baseline_ratio: int
    modulation_period: Any
    baseline: Any = None
    type: Any = None

    @field_validator('baseline_minimum')
    def validate_baseline_minimum(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the baseline minimum input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The minimum baseline in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)

    @field_validator('baseline_maximum')
    def validate_baseline_maximum(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the baseline maximum input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The maximum baseline in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)

    @field_validator('modulation_period')
    def validate_modulation_period(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the modulation period input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The modulation period in units of time"
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.s)

    @abstractmethod
    def get_collector_positions(self, time: astropy.units.Quantity) -> np.ndarray:
        """Return an array containing the time-dependent x- and y-coordinates of the collectors.
        :param time: Time variable in seconds
        :return: An array containing the coordinates.
        """
        pass

    @abstractmethod
    def get_optimal_baseline(self, wavelength: astropy.units.Quantity,
                             optimal_angular_distance: astropy.units.Quantity) -> astropy.units.Quantity:
        """Return the optimal baseline for a given wavelength and a angular separation that is to be optimized for

        :param wavelength: Wavelength to be optimized for
        :param optimal_angular_distance: Angular distance between the star and a (potential) planet to be optimized for
        :return: The baseline
        """
        pass

    def set_optimal_baseline(self, target_systems: list[dict], optimized_wavelength: astropy.units.Quantity):
        """Set the baseline to optimize for the habitable zone, if it is between the minimum and maximum allowed
        baselines.
        """
        for target_system in target_systems:
            star = [value for value in target_system.values() if isinstance(value, Star)][0]
            optimal_baseline = self.get_optimal_baseline(wavelength=optimized_wavelength,
                                                         optimal_angular_distance=star.habitable_zone_central_angular_radius).to(
                u.m)

            if self.baseline_minimum <= optimal_baseline and optimal_baseline <= self.baseline_maximum:
                self.baseline = optimal_baseline
            else:
                raise ValueError(
                    f'Optimal baseline of {optimal_baseline} is not within allowed ranges of baselines {self.baseline_minimum}-{self.baseline_maximum}')


class EmmaXCircularRotation(ArrayConfiguration):
    """Class representation of the Emma-X array configuration with circular rotation of the array.
    """
    type: Any = ArrayConfigurationEnum.EMMA_X_CIRCULAR_ROTATION

    def get_collector_positions(self, time: astropy.units.Quantity) -> np.ndarray:
        rotation_matrix = get_2d_rotation_matrix(time, self.modulation_period)
        emma_x_static = self.baseline / 2 * np.array(
            [[self.baseline_ratio, self.baseline_ratio, -self.baseline_ratio, -self.baseline_ratio], [1, -1, -1, 1]])
        return np.matmul(rotation_matrix, emma_x_static)

    def get_optimal_baseline(self, wavelength: astropy.units.Quantity,
                             optimal_angular_distance: astropy.units.Quantity):
        return 0.59 * wavelength.to(u.m) / optimal_angular_distance.to(u.rad) * u.rad


class EmmaXDoubleStretch(ArrayConfiguration):
    """Class representation of the Emma-X array configuration with double stretching of the array.
    """
    type: Any = ArrayConfigurationEnum.EMMA_X_DOUBLE_STRETCH

    def get_collector_positions(self, time: float) -> np.ndarray:
        emma_x_static = self.baseline / 2 * np.array(
            [[self.baseline_ratio, self.baseline_ratio, -self.baseline_ratio, -self.baseline_ratio], [1, -1, -1, 1]])
        # TODO: fix calculations
        return emma_x_static * (1 + (2 * self.baseline) / self.baseline * np.sin(
            2 * np.pi * u.rad / self.modulation_period * time))

    def get_optimal_baseline(self, wavelength: astropy.units.Quantity,
                             optimal_angular_distance: astropy.units.Quantity):
        # TODO: Implement correct formula
        return 0.59 * wavelength / optimal_angular_distance.to(u.rad) * u.rad


class EquilateralTriangleCircularRotation(ArrayConfiguration):
    """Class representation of an equilateral triangle configuration with circular rotation of the array.
    """
    type: Any = ArrayConfigurationEnum.EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION

    def get_collector_positions(self, time: float) -> np.ndarray:
        height = np.sqrt(3) / 2 * self.baseline
        height_to_center = height / 3
        rotation_matrix = get_2d_rotation_matrix(time, self.modulation_period)

        equilateral_triangle_static = np.array(
            [[0, self.baseline.value / 2, -self.baseline.value / 2],
             [height.value - height_to_center.value, -height_to_center.value, -height_to_center.value]])

        return np.matmul(rotation_matrix, equilateral_triangle_static) * self.baseline.unit

    def get_optimal_baseline(self, wavelength: astropy.units.Quantity,
                             optimal_angular_distance: astropy.units.Quantity):
        # TODO: Implement correct formula
        return 0.59 * wavelength / optimal_angular_distance.to(u.rad) * u.rad


class RegularPentagonCircularRotation(ArrayConfiguration):
    """Class representation of a regular pentagon configuration with circular rotation of the array.
    """
    type: Any = ArrayConfigurationEnum.REGULAR_PENTAGON_CIRCULAR_ROTATION

    def _x(self, angle) -> astropy.units.Quantity:
        """Return the x position.

        :param angle: The angle at which the collector is located
        :return: The x position
        """
        return 0.851 * self.baseline.value * np.cos(angle)

    def _y(self, angle) -> astropy.units.Quantity:
        """Return the y position.

        :param angle: The angle at which the collector is located
        :return: The y position
        """
        return 0.851 * self.baseline.value * np.sin(angle)

    def get_collector_positions(self, time: float) -> np.ndarray:
        angles = [0, 2 * np.pi / 5, 4 * np.pi / 5, 6 * np.pi / 5, 8 * np.pi / 5]
        rotation_matrix = get_2d_rotation_matrix(time, self.modulation_period)
        pentagon_static = np.array([
            [self._x(angles[0]), self._x(angles[1]), self._x(angles[2]), self._x(angles[3]), self._x(angles[4])],
            [self._y(angles[0]), self._y(angles[1]), self._y(angles[2]), self._y(angles[3]), self._y(angles[4])]])
        return np.matmul(rotation_matrix, pentagon_static) * self.baseline.unit

    def get_optimal_baseline(self, wavelength: astropy.units.Quantity,
                             optimal_angular_distance: astropy.units.Quantity):
        # TODO: Implement correct formula
        return 0.59 * wavelength / optimal_angular_distance.to(u.rad) * u.rad
