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

    @property
    def solid_angle(self):
        return np.pi * (((self.radius.to(u.m)) ** 2 / (self.distance.to(u.m))) * u.rad) ** 2

    @property
    def angular_radius(self):
        return ((self.radius.to(u.m) / self.distance.to(u.m)) * u.rad).to(u.arcsec)

    def get_sky_coordinate_maps(self) -> Tuple:
        return get_meshgrid(2 * (1.05 * self.angular_radius), self.grid_size)

    def get_shape_map(self) -> np.ndarray:
        return (np.sqrt(self.sky_coordinate_maps[0] ** 2 + self.sky_coordinate_maps[1] ** 2) <= self.angular_radius)
