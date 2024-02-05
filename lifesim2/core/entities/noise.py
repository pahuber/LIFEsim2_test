from typing import Any, Optional

import astropy
import colorednoise as cn
import numpy as np
from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from lifesim2.core.processing.noise_generator import NoiseGenerator
from lifesim2.io.validators import validate_quantity_units


class PhasePerturbations(BaseModel):
    apply: bool
    power_law_exponent: int
    rms: Any

    @field_validator('rms')
    def _validate_rms(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the rms input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The rms in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))


class PolarizationPerturbations(BaseModel):
    apply: bool
    power_law_exponent: int
    rms: Any

    @field_validator('rms')
    def _validate_rms(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the rms input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The rms in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.rad,))


class Noise(BaseModel):
    """Class representation of the noise contributions that are considered for the simulation.
    """
    stellar_leakage: bool
    local_zodi_leakage: bool
    exozodi_leakage: bool
    amplitude_perturbations: bool
    phase_perturbations: Optional[PhasePerturbations]
    polarization_perturbations: Optional[PolarizationPerturbations]
    phase_perturbation_distribution: Any = None
    polarization_perturbation_distribution: Any = None

    def get_perturbation_distribution(self,
                                      number_of_inputs: int,
                                      sample_time: astropy.units.Quantity,
                                      number_of_samples: int,
                                      rms: astropy.units.Quantity,
                                      color_exponent: int) -> np.ndarray:
        """Return a distribution that phase differences should be drawn from. The distribution is created using a power
        law as 1/f^exponent.

        :param time_step: The time step used to calculate the maximum frequency
        :return: The distribution
        """
        noise_generator = NoiseGenerator()
        perturbation_distributions = np.zeros((number_of_inputs, number_of_samples)) * rms.unit
        sample_time = 1 / 10000  # Corresponds to 10kHz

        # TODO: make perturbation distributions chromatic and use correct cutoff frequency
        for i in range(number_of_inputs):
            match color_exponent:
                case 0:
                    perturbation_distributions[i] = noise_generator.generate(dt=sample_time,
                                                                             n=number_of_samples,
                                                                             colour=noise_generator.white()) * rms.unit
                case 1:
                    perturbation_distributions[i] = noise_generator.generate(dt=sample_time,
                                                                             n=number_of_samples,
                                                                             colour=noise_generator.pink()) * rms.unit
                case 2:
                    perturbation_distributions[i] = noise_generator.generate(dt=sample_time,
                                                                             n=number_of_samples,
                                                                             colour=noise_generator.brown()) * rms.unit

            perturbation_distributions[i] *= rms.value / np.sqrt(
                np.mean(perturbation_distributions[i] ** 2)) * rms.unit
        return perturbation_distributions

    def get_phase_perturbations_distribution(self, time_step: astropy.units.Quantity) -> np.ndarray:
        """Return a distribution that phase differences should be drawn from. The distribution is created using a power
        law as 1/f^exponent.

        :param time_step: The time step used to calculate the maximum frequency
        :return: The distribution
        """
        phase_difference_distribution = cn.powerlaw_psd_gaussian(1, 1000, 1 / time_step.to(u.s).value)
        phase_difference_distribution *= self.phase_perturbations.rms.value / np.sqrt(
            np.mean(phase_difference_distribution ** 2))
        self.phase_perturbation_distribution = phase_difference_distribution * self.phase_perturbations.rms.unit
