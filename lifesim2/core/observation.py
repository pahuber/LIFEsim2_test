from typing import Any

import astropy.units
from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from lifesim2.io.validators import validate_quantity_units


class Observation(BaseModel):
    """Class representation of an observation."""

    adjust_baseline_to_habitable_zone: bool
    integration_time: Any
    optimized_wavelength: Any
    x_sky_coordinates_map: Any = None
    y_sky_coordinates_map: Any = None
    observatory: Any = None
    sources: list = []

    @field_validator('integration_time')
    def validate_integration_time(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.s)

    @field_validator('optimized_wavelength')
    def validate_optimized_wavelength(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)
