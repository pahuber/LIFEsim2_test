from typing import Any, Tuple

import astropy
import numpy as np
from astropy import units as u
from astropy.constants.codata2018 import G
from poliastro.bodies import Body
from poliastro.twobody import Orbit
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo

from sygn.core.module.target_system.source import Source
from sygn.io.validators import validate_quantity_units
from sygn.util.grid import get_index_of_closest_value, get_meshgrid


class Planet(Source):
    """Class representation of a planet.
    """
    name: str
    temperature: Any
    radius: Any
    mass: Any
    semi_major_axis: Any
    eccentricity: float
    inclination: Any
    raan: Any
    argument_of_periapsis: Any
    true_anomaly: Any
    star_distance: Any
    star_mass: Any
    number_of_wavelength_bins: int
    grid_size: int
    mean_spectral_flux_density: Any = None
    position_x: Any = None
    position_y: Any = None

    @field_validator('argument_of_periapsis')
    def _validate_argument_of_periapsis(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the argument of periapsis input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The argument of periapsis in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.deg)

    @field_validator('inclination')
    def _validate_inclination(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the inclination input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The inclination in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.deg)

    @field_validator('mass')
    def _validate_mass(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the mass input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The mass in units of weight
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.kg)

    @field_validator('raan')
    def _validate_raan(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the raan input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The raan in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.deg)

    @field_validator('radius')
    def _validate_radius(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the radius input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The radius in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)

    @field_validator('semi_major_axis')
    def _validate_semi_major_axis(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the semi-major axis input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The semi-major axis in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)

    @field_validator('true_anomaly')
    def _validate_true_anomaly(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the true anomaly input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The true anomaly in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.deg)

    @property
    def solid_angle(self) -> astropy.units.Quantity:
        """Return the solid angle covered by the planet on the sky.

        :return: The solid angle
        """
        return np.pi * (self.radius.to(u.m) / (self.star_distance.to(u.m)) * u.rad) ** 2

    # @property
    # def star_angular_separation_x(self) -> astropy.units.Quantity:
    #     """Return the angular star separation x.
    #
    #     :return: The angular star separation x
    #     """
    #     return ((self.star_separation_x.to(u.m) / self.star_distance.to(u.m)) * u.rad).to(u.arcsec)
    #
    # @property
    # def star_angular_separation_y(self) -> astropy.units.Quantity:
    #     """Return the angular star separation y.
    #
    #     :return: The angular star separation y
    #     """
    #     return ((self.star_separation_y.to(u.m) / self.star_distance.to(u.m)) * u.rad).to(u.arcsec)

    def get_sky_coordinate_maps(self, time: astropy.units.Quantity) -> Tuple[np.ndarray, np.ndarray]:
        """Return the sky coordinate maps of the source. The intensity responses are calculated in a resolution that
        allows the source to fill the grid, thus, each source needs to define its own sky coordinate map. Add 10% to the
        angular radius to account for rounding issues and make sure the source is fully covered within the map.

        :return: A tuple containing the x- and y-sky coordinate maps
        """
        angular_separation_from_star_x, angular_separation_from_star_y = self._get_x_y_angular_separation_from_star(
            time)
        if np.abs(angular_separation_from_star_x) >= np.abs(angular_separation_from_star_y):
            return get_meshgrid(2 * (1.05 * np.abs(angular_separation_from_star_x)), self.grid_size)
        return get_meshgrid(2 * (1.05 * np.abs(angular_separation_from_star_y)), self.grid_size)

    def get_sky_position_map(self, time: astropy.units.Quantity) -> np.ndarray:
        sky_coordinate_maps = self.get_sky_coordinate_maps(time)
        position_map = np.zeros(sky_coordinate_maps[0].shape)
        angular_separation_from_star_x, angular_separation_from_star_y = self._get_x_y_angular_separation_from_star(
            time)
        index_x = get_index_of_closest_value(sky_coordinate_maps[0][0, :], angular_separation_from_star_x)
        index_y = get_index_of_closest_value(sky_coordinate_maps[1][:, 0], angular_separation_from_star_y)
        position_map[index_y][index_x] = 1
        return position_map

    def _get_x_y_separation_from_star(self, time: astropy.units.Quantity) -> Tuple:
        """Return the separation of the planet from the star in x- and y-direction.

        :param time: The time
        :return: A tuple containing the x- and y- coordinates
        """
        star = Body(parent=None, k=G * (self.star_mass + self.mass), name='Star')
        orbit = Orbit.from_classical(star, a=self.semi_major_axis, ecc=u.Quantity(self.eccentricity),
                                     inc=self.inclination,
                                     raan=self.raan,
                                     argp=self.argument_of_periapsis, nu=self.true_anomaly)
        orbit_propagated = orbit.propagate(time)
        return (orbit_propagated.r[0], orbit_propagated.r[1])

    def _get_x_y_angular_separation_from_star(self, time: astropy.units.Quantity) -> Tuple:
        """Return the angular separation of the planet from the star in x- and y-direction.

        :param time: The time
        :return: A tuple containing the x- and y- coordinates
        """
        separation_from_star_x, separation_from_star_y = self._get_x_y_separation_from_star(time)
        angular_separation_from_star_x = ((separation_from_star_x.to(u.m) / self.star_distance.to(u.m)) * u.rad).to(
            u.arcsec)
        angular_separation_from_star_y = ((separation_from_star_y.to(u.m) / self.star_distance.to(u.m)) * u.rad).to(
            u.arcsec)
        return (angular_separation_from_star_x, angular_separation_from_star_y)
