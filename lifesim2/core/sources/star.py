from typing import Any

import astropy
import numpy as np
from astropy import units as u
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo

from lifesim2.core.sources.source import Source
from lifesim2.io.validators import validate_quantity_units


class Star(Source):
    name: str
    temperature: Any
    radius: Any
    mass: Any
    distance: Any
    number_of_wavelength_bins: int
    position: Any = (0, 0) * u.arcsec

    @field_validator('radius')
    def validate_radius(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)

    @property
    @field_validator('mass')
    def validate_mass(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.kg)

    @field_validator('distance')
    def validate_distance(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)

    def solid_angle(self):
        return np.pi * (((self.radius.to(u.m)) ** 2 / (self.distance.to(u.m))) * u.rad) ** 2
