from typing import Any, Tuple

import astropy
import numpy as np
from astropy import units as u
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo

from lifesim2.core.sources.source import Source
from lifesim2.io.validators import validate_quantity_units
from lifesim2.util.grid import get_meshgrid


class Star(Source):
    name: str
    temperature: Any
    radius: Any
    mass: Any
    distance: Any
    luminosity: Any
    number_of_wavelength_bins: int
    grid_size: int

    @field_validator('radius')
    def validate_radius(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)

    @field_validator('mass')
    def validate_mass(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.kg)

    @field_validator('distance')
    def validate_distance(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)

    @field_validator('luminosity')
    def validate_luminosity(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.Lsun)

    @property
    def solid_angle(self):
        return np.pi * (((self.radius.to(u.m)) ** 2 / (self.distance.to(u.m))) * u.rad) ** 2

    @property
    def angular_radius(self):
        return ((self.radius.to(u.m) / self.distance.to(u.m)) * u.rad).to(u.arcsec)

    @property
    def habitable_zone_central_radius(self):
        s0_in, s0_out = 1.7665, 0.3240
        a_in, a_out = 1.3351E-4, 5.3221E-5
        b_in, b_out = 3.1515E-9, 1.4288E-9
        c_in, c_out = -3.3488E-12, -1.1049E-12
        t = self.temperature.value - 5780

        s_eff_in = s0_in + a_in * t + b_in * t ** 2 + c_in * t ** 3
        s_eff_out = s0_out + a_out * t + b_out * t ** 2 + c_out * t ** 3

        r_in = np.sqrt(self.luminosity.value / s_eff_in)
        r_out = np.sqrt(self.luminosity.value / s_eff_out)
        return ((r_out + r_in) / 2 * u.au).to(u.m)

    @property
    def habitable_zone_central_angular_radius(self):
        return (self.habitable_zone_central_radius / self.distance * u.rad).to(u.arcsec)

    def get_sky_coordinate_maps(self) -> Tuple:
        return get_meshgrid(2 * (1.05 * self.angular_radius), self.grid_size)

    def get_shape_map(self) -> np.ndarray:
        return (np.sqrt(self.sky_coordinate_maps[0] ** 2 + self.sky_coordinate_maps[1] ** 2) <= self.angular_radius)
