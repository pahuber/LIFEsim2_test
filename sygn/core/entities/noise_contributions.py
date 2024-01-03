from typing import Any, Optional

import astropy
import colorednoise as cn
import numpy as np
from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from sygn.io.validators import validate_quantity_units


class OpticalPathDifferenceVariability(BaseModel):
    apply: bool
    power_law_exponent: int
    rms: Any

    @field_validator('rms')
    def _validate_optimized_wavelength(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the optimized wavelength input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The rms in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))


class NoiseContributions(BaseModel):
    """Class representation of the noise contributions that are considered for the simulation.
    """
    stellar_leakage: bool
    local_zodi_leakage: bool
    exozodi_leakage: bool
    fiber_injection_variability: bool
    optical_path_difference_variability: Optional[OpticalPathDifferenceVariability]
    optical_path_difference_distribution: Any = None

    def get_optical_path_difference_distribution(self, time_step: astropy.units.Quantity) -> np.ndarray:
        """Return a distribution that phase differences should be drawn from. The distribution is created using a power
        law as 1/f^exponent.

        :param time_step: The time step used to calculate the maximum frequency
        :return: The distribution
        """
        phase_difference_distribution = cn.powerlaw_psd_gaussian(1, 1000, 1 / time_step.to(u.s).value)
        phase_difference_distribution *= self.optical_path_difference_variability.rms.value / np.sqrt(
            np.mean(phase_difference_distribution ** 2))
        self.optical_path_difference_distribution = phase_difference_distribution * self.optical_path_difference_variability.rms.unit
