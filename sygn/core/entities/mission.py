from typing import Any

import astropy.units
from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from sygn.io.validators import validate_quantity_units


class Mission(BaseModel):
    """Class representation of a mission."""
    integration_time: Any
    modulation_period: Any
    baseline_minimum: Any
    baseline_maximum: Any
    baseline_ratio: int
    optimized_differential_output: int
    optimized_star_separation: Any
    optimized_wavelength: Any

    @field_validator('baseline_minimum')
    def _validate_baseline_minimum(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the baseline minimum input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The minimum baseline in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @field_validator('baseline_maximum')
    def _validate_baseline_maximum(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the baseline maximum input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The maximum baseline in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @field_validator('integration_time')
    def _validate_integration_time(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the integration time input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The integration time in units of time
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.s,))

    @field_validator('modulation_period')
    def _validate_modulation_period(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the modulation period input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The modulation period in units of time
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.s,))

    @field_validator('optimized_star_separation')
    def _validate_optimized_star_separation(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the optimized star separation input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The optimized star separation in its original units or as a string
        """
        if value == 'habitable-zone':
            return value
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m, u.arcsec))

    @field_validator('optimized_wavelength')
    def _validate_optimized_wavelength(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the optimized wavelength input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The optimized wavelength in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))
