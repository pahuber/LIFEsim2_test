from typing import Any, Tuple

import astropy
import numpy as np
from astropy import units as u
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo

from lifesim2.core.sources.source import Source
from lifesim2.io.validators import validate_quantity_units
from lifesim2.util.blackbody import create_blackbody_spectrum
from lifesim2.util.grid import get_meshgrid


class Star(Source):
    """Class representation of a star.
    """
    name: str
    temperature: Any
    radius: Any
    mass: Any
    distance: Any
    luminosity: Any
    zodi_level: int
    wavelength_range_lower_limit: Any
    wavelength_range_upper_limit: Any
    wavelength_bin_centers: Any
    wavelength_bin_widths: Any
    grid_size: int

    def __init__(self, **data):
        """Constructor method.

        :param data: Data to initialize the star class.
        """
        super().__init__(**data)
        self.flux = create_blackbody_spectrum(self.temperature,
                                              self.wavelength_range_lower_limit,
                                              self.wavelength_range_upper_limit,
                                              self.wavelength_bin_centers,
                                              self.wavelength_bin_widths,
                                              self.solid_angle)

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

    @field_validator('distance')
    def validate_distance(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the distance input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The distance in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)

    @field_validator('luminosity')
    def validate_luminosity(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the luminosity input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The luminosity in units of luminosity
        """
        # TODO: force Lsun units
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.Lsun)

    @property
    def solid_angle(self) -> astropy.units.Quantity:
        """Return the solid angle that the source object covers on the sky.

        :return: The solid angle
        """
        return np.pi * (self.radius.to(u.m) / (self.distance.to(u.m)) * u.rad) ** 2

    @property
    def angular_radius(self) -> astropy.units.Quantity:
        """Return the solid angle covered by the star on the sky.

        :return: The solid angle
        """
        return ((self.radius.to(u.m) / self.distance.to(u.m)) * u.rad).to(u.arcsec)

    @property
    def habitable_zone_central_radius(self) -> astropy.units.Quantity:
        """Return the central habitable zone radius of the star. Calculated as defined in Kopparapu et al. 2013.

        :return: The central habitable zone radius
        """
        incident_solar_flux_inner, incident_solar_flux_outer = 1.7665, 0.3240
        parameter_a_inner, parameter_a_outer = 1.3351E-4, 5.3221E-5
        parameter_b_inner, parameter_b_outer = 3.1515E-9, 1.4288E-9
        parameter_c_inner, parameter_c_outer = -3.3488E-12, -1.1049E-12
        temperature_difference = self.temperature.value - 5780

        incident_stellar_flux_inner = (incident_solar_flux_inner + parameter_a_inner * temperature_difference
                                       + parameter_b_inner * temperature_difference ** 2 + parameter_c_inner
                                       * temperature_difference ** 3)
        incident_stellar_flux_outer = (incident_solar_flux_outer + parameter_a_outer * temperature_difference
                                       + parameter_b_outer * temperature_difference ** 2 + parameter_c_outer
                                       * temperature_difference ** 3)

        radius_inner = np.sqrt(self.luminosity.value / incident_stellar_flux_inner)
        radius_outer = np.sqrt(self.luminosity.value / incident_stellar_flux_outer)
        return ((radius_outer + radius_inner) / 2 * u.au).to(u.m)

    @property
    def habitable_zone_central_angular_radius(self) -> astropy.units.Quantity:
        """Return the central habitable zone radius in angular units.

        :return: The central habitable zone radius in angular units
        """
        return (self.habitable_zone_central_radius / self.distance * u.rad).to(u.arcsec)

    def get_sky_coordinate_maps(self) -> Tuple:
        return get_meshgrid(2 * (1.05 * self.angular_radius), self.grid_size)

    def get_shape_map(self) -> np.ndarray:
        return (np.sqrt(self.sky_coordinate_maps[0] ** 2 + self.sky_coordinate_maps[1] ** 2) <= self.angular_radius)
