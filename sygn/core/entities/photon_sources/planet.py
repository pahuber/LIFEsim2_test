from typing import Any, Tuple

import astropy
import numpy as np
from astropy import units as u
from astropy.constants.codata2018 import G
from poliastro.bodies import Body
from poliastro.twobody import Orbit
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo

from sygn.core.context import Context
from sygn.core.entities.photon_sources.photon_source import PhotonSource
from sygn.io.validators import validate_quantity_units
from sygn.util.blackbody import create_blackbody_spectrum
from sygn.util.grid import get_index_of_closest_value, get_meshgrid
from sygn.util.helpers import Coordinates


class Planet(PhotonSource):
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
    number_of_wavelength_bins: int = None
    grid_size: int = None
    # mean_spectral_flux_density: Any = None
    position_x: Any = None
    position_y: Any = None
    angular_separation_from_star_x: Any = None
    angular_separation_from_star_y: Any = None

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

    @field_validator('temperature')
    def _validate_temperature(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the temperature input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The temperature in units of temperature
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.K)

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

    def _get_x_y_separation_from_star(self, time: astropy.units.Quantity, planet_orbital_motion: bool) -> Tuple:
        """Return the separation of the planet from the star in x- and y-direction. If the planet orbital motion is
        considered, calculate the new position for each time step.

        :param time: The time
        :param planet_orbital_motion: Whether the planet orbital motion is to be considered
        :return: A tuple containing the x- and y- coordinates
        """
        star = Body(parent=None, k=G * (self.star_mass + self.mass), name='Star')
        orbit = Orbit.from_classical(star, a=self.semi_major_axis, ecc=u.Quantity(self.eccentricity),
                                     inc=self.inclination,
                                     raan=self.raan,
                                     argp=self.argument_of_periapsis, nu=self.true_anomaly)
        if planet_orbital_motion:
            orbit_propagated = orbit.propagate(time)
        else:
            orbit_propagated = orbit
        return (orbit_propagated.r[0], orbit_propagated.r[1])

    def _get_x_y_angular_separation_from_star(self, time: astropy.units.Quantity, planet_orbital_motion: bool) -> Tuple:
        """Return the angular separation of the planet from the star in x- and y-direction.

        :param time: The time
        :param planet_orbital_motion: Whether the planet orbital motion is to be considered
        :return: A tuple containing the x- and y- coordinates
        """
        separation_from_star_x, separation_from_star_y = self._get_x_y_separation_from_star(time, planet_orbital_motion)
        angular_separation_from_star_x = ((separation_from_star_x.to(u.m) / self.star_distance.to(u.m)) * u.rad).to(
            u.arcsec)
        angular_separation_from_star_y = ((separation_from_star_y.to(u.m) / self.star_distance.to(u.m)) * u.rad).to(
            u.arcsec)
        return (angular_separation_from_star_x, angular_separation_from_star_y)

    def _calculate_sky_coordinates(self, context: Context) -> np.ndarray:
        """Calculate and return the sky coordinates of the source. Add 40% to the angular radius to account for rounding
        issues and make sure the source is fully covered within the map.

        :param context: Context
        :return: The sky coordinates
        """
        sky_coordinates = np.zeros((len(context.time_range_planet_motion)), dtype=object)

        for index_time, time in enumerate(context.time_range_planet_motion):
            self.angular_separation_from_star_x, self.angular_separation_from_star_y = self._get_x_y_angular_separation_from_star(
                time, context.settings.planet_orbital_motion)

            # Depending on whether the x- or y-angular separation of planet is larger, the respective value is taken as
            # the maximum extent of the grid
            if np.abs(self.angular_separation_from_star_x) >= np.abs(self.angular_separation_from_star_y):
                sky_coordinates_at_time_step = get_meshgrid(2 * (1.2 * np.abs(self.angular_separation_from_star_x)),
                                                            context.settings.grid_size)
            else:
                sky_coordinates_at_time_step = get_meshgrid(2 * (1.2 * np.abs(self.angular_separation_from_star_y)),
                                                            context.settings.grid_size)
            sky_coordinates[index_time] = Coordinates(sky_coordinates_at_time_step[0], sky_coordinates_at_time_step[1])

        return sky_coordinates

    def _calculate_sky_brightness_distribution(self, context: Context) -> np.ndarray:
        """Calculate and return the sky brightness distribution.

        :param context: The context
        :return: The sky brightness distribution
        """
        sky_brightness_distribution = np.zeros(
            ((len(self.sky_coordinates), len(self.mean_spectral_flux_density),) + self.sky_coordinates[0].x.shape)) * \
                                      self.mean_spectral_flux_density[0].unit

        for index_sky_coordinates, sky_coordinates in enumerate(self.sky_coordinates):
            # Find the index corresponding to the value of the sky coordinates that matches the closest to the position
            # of the planet on the sky
            index_x = get_index_of_closest_value(sky_coordinates.x[0, :], self.angular_separation_from_star_x)
            index_y = get_index_of_closest_value(sky_coordinates.y[:, 0], self.angular_separation_from_star_y)
            for index_wavelength in range(len(self.mean_spectral_flux_density)):
                sky_brightness_distribution[index_sky_coordinates][index_wavelength][index_y][index_x] = \
                    self.mean_spectral_flux_density[index_wavelength]
        return sky_brightness_distribution

    def _calculate_mean_spectral_flux_density(self, context: Context) -> np.ndarray:
        """Calculate the mean spectral flux density of the planet.

        :param context: The context
        :return: The mean spectral flux density
        """
        return create_blackbody_spectrum(self.temperature,
                                         context.observatory.instrument_parameters.wavelength_range_lower_limit,
                                         context.observatory.instrument_parameters.wavelength_range_upper_limit,
                                         context.observatory.instrument_parameters.wavelength_bin_centers,
                                         context.observatory.instrument_parameters.wavelength_bin_widths,
                                         self.solid_angle)

    def get_sky_coordinates(self, index_time: int, index_wavelength: int) -> Coordinates:
        """Return the sky coordinates for a given time index. The coordinates are wavelength-independent and only time-
        dependent, if its orbital motion is considered.

        :param index_time: The time index
        :param index_wavelength: The wavelength index
        :return: The sky coordinates
        """
        return self.sky_coordinates[index_time]

    def get_sky_brightness_distribution(self, index_time: int, index_wavelength: int) -> np.ndarray:
        """Return the sky brightness distribution for the planet, which is both time- and wavelength-dependent, if its
        orbital motion is considered.

        :param index_time: The time index
        :param index_wavelength: The wavelength index
        :return: The sky brightness distribution
        """
        return self.sky_brightness_distribution[index_time][index_wavelength]
