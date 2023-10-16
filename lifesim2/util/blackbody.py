import astropy.units
import numpy as np
import spectres
from astropy import units as u
from astropy.modeling.models import BlackBody


def create_blackbody_spectrum(temperature,
                              wavelength_range_lower_limit: astropy.units.Quantity,
                              wavelength_range_upper_limit: astropy.units.Quantity,
                              wavelength_bin_centers: np.ndarray,
                              wavelength_bin_widths: np.ndarray,
                              angle_per_pixel: astropy.units.Quantity) -> np.ndarray:
    """Return a blackbody spectrum for an astrophysical object. The spectrum is binned already to the wavelength bin
    centers of the observation.

    :param temperature: Temperature of the astrophysical object
    :param wavelength_range_lower_limit: Lower limit of the wavelength range
    :param wavelength_range_upper_limit: Upper limit of the wavelength range
    :param wavelength_bin_centers: Array containing the wavelength bin centers
    :param wavelength_bin_widths: Array containing the wavelength bin widths
    :param angle_per_pixel: Angle per grid pixel of the sky coordinates map
    :return: Array containing the flux per bin in units of ph m-2 s-1 um-1
    """
    wavelength_range = np.linspace(wavelength_range_lower_limit.value, wavelength_range_upper_limit.value,
                                   100) * wavelength_range_upper_limit.unit

    blackbody_spectrum = BlackBody(temperature=temperature)(wavelength_range)

    units = blackbody_spectrum.unit

    blackbody_spectrum_binned = spectres.spectres(new_wavs=wavelength_bin_centers.to(u.um).value,
                                                  spec_wavs=wavelength_range.to(u.um).value,
                                                  spec_fluxes=blackbody_spectrum.value) * units

    return convert_blackbody_to_flux(blackbody_spectrum_binned, angle_per_pixel, wavelength_bin_centers)


def convert_blackbody_to_flux(blackbody_spectrum_binned: np.ndarray,
                              angle_per_pixel: astropy.units.Quantity,
                              wavelength_bin_centers: np.ndarray) -> np.ndarray:
    """Convert the binned black body spectrum from units erg / (Hz s sr cm2) to units ph / (m2 s um)

    :param blackbody_spectrum_binned: The binned blackbody spectrum
    :param angle_per_pixel: Angle per grid pixel of the sky coordinates map
    :param wavelength_bin_centers: The wavelength bin centers
    :return: Array containing the spectrum in correct units
    """
    flux = np.zeros(len(blackbody_spectrum_binned)) * u.ph / u.m ** 2 / u.s / u.um

    for index in range(len(blackbody_spectrum_binned)):
        current_flux = (blackbody_spectrum_binned[index] * (angle_per_pixel[index] ** 2).to(u.sr)).to(
            u.ph / u.m ** 2 / u.s / u.um,
            equivalencies=u.spectral_density(
                wavelength_bin_centers[index]))
        # TODO: fix flux too low
        flux[index] = current_flux
    return flux
