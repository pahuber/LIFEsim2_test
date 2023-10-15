import astropy.units
import numpy as np
from astropy import units as u
from astropy.modeling.models import BlackBody

from lifesim2.core.sources.source import Source


def create_blackbody_spectrum(temperature,
                              spectral_range_lower_limit: astropy.units.Quantity,
                              spectral_range_upper_limit: astropy.units.Quantity):
    wavelength_range = np.linspace(spectral_range_lower_limit.value, spectral_range_upper_limit.value,
                                   Source.number_of_steps) * spectral_range_upper_limit.unit

    blackbody_spectrum = BlackBody(temperature=temperature)(wavelength_range)

    return convert_blackbody_to_flux(blackbody_spectrum, wavelength_range)


def convert_blackbody_to_flux(blackbody_spectrum, wavelength_range):
    flux = np.zeros(Source.number_of_steps) * u.ph / u.m ** 2 / u.s / u.um

    for index in range(len(blackbody_spectrum)):
        # TODO: fix sr calculation
        current_flux = (blackbody_spectrum[index] * u.sr).to(u.ph / u.m ** 2 / u.s / u.um,
                                                             equivalencies=u.spectral_density(
                                                                 wavelength_range[index]))
        flux[index] = current_flux
    return flux
