from typing import Any, Optional

import astropy
from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from lifesim2.io.validators import validate_quantity_units


class OpticalPathDifferenceVariability(BaseModel):
    apply: bool
    power_law_exponent: int
    rms: Any

    @field_validator('rms')
    def validate_optimized_wavelength(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the optimized wavelength input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The rms in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)


class NoiseContributions(BaseModel):
    """Class representation of the noise contributions that are considered for the simulation.
    """
    stellar_leakage: bool
    local_zodi_leakage: bool
    exozodi_leakage: bool
    fiber_injection_variability: bool
    optical_path_difference_variability: Optional[OpticalPathDifferenceVariability]
