from typing import Any, Tuple

import astropy
import numpy as np
from astropy import units as u
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo

from lifesim2.core.sources.source import Source
from lifesim2.io.validators import validate_quantity_units
from lifesim2.util.grid import get_index_of_closest_value, get_meshgrid


class Planet(Source):
    name: str
    temperature: Any
    radius: Any
    mass: Any
    star_separation: Any
    star_distance: Any
    number_of_wavelength_bins: int
    grid_size: int

    @field_validator('radius')
    def validate_radius(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the radius input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The radius in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)

    @field_validator('mass')
    def validate_mass(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the mass input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The mass in units of weight
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.kg)

    @field_validator('star_separation')
    def validate_star_separation(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the star separation input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The star separation in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)

    @field_validator('star_distance')
    def validate_star_distance(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the star distance input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The star distance in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)

    @property
    def solid_angle(self):
        """Return the solid angle filled by the planet.

        :return: The solid angle
        """
        return np.pi * (((self.radius.to(u.m)) ** 2 / (self.distance.to(u.m))) * u.rad) ** 2

    @property
    def star_angular_separation(self):
        """Return the angular star separation.

        :return: The angular star separation
        """
        return ((self.star_separation.to(u.m) / self.star_distance.to(u.m)) * u.rad).to(u.arcsec)

    def get_sky_coordinate_maps(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the x- and y-sky-coordinate maps. Extend the coordinates by 10% of the planet star separation.

        :return: A tuple containing the x- and y-sky coordinate maps
        """
        return get_meshgrid(2 * (1.05 * self.star_angular_separation), self.grid_size)

    def get_shape_map(self) -> np.ndarray:
        """Return the shape map of the planets. Consists of zero everywhere, but at the planet position, where it is one.

        :return: The shape map
        """
        position_map = np.zeros(self.sky_coordinate_maps[0].shape)

        # TODO: implement planet position correctly
        x = self.star_angular_separation * np.cos(0)
        y = self.star_angular_separation * np.sin(0)

        index_x = get_index_of_closest_value(self.sky_coordinate_maps[0][0, :], x)
        index_y = get_index_of_closest_value(self.sky_coordinate_maps[1][:, 0], y)

        position_map[index_y][index_x] = 1
        return position_map
