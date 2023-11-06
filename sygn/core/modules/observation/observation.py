from typing import Any

import astropy.units
from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from sygn.io.validators import validate_quantity_units


class Observation(BaseModel):
    """Class representation of an observation."""
    adjust_baseline_to_habitable_zone: bool
    integration_time: Any
    optimized_wavelength: Any

    @field_validator('integration_time')
    def validate_integration_time(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the integration time input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The integration time in units of time
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.s)

    @field_validator('optimized_wavelength')
    def validate_optimized_wavelength(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the optimized wavelength input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The optimized wavelength in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)
