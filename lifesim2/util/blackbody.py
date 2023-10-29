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
                              source_solid_angle: astropy.units.Quantity) -> np.ndarray:
    """Return a blackbody spectrum for an astrophysical object. The spectrum is binned already to the wavelength bin
    centers of the observation.

    :param temperature: Temperature of the astrophysical object
    :param wavelength_range_lower_limit: Lower limit of the wavelength range
    :param wavelength_range_upper_limit: Upper limit of the wavelength range
    :param wavelength_bin_centers: Array containing the wavelength bin centers
    :param wavelength_bin_widths: Array containing the wavelength bin widths
    :param source_solid_angle: The solid angle of the source
    :return: Array containing the flux per bin in units of ph m-2 s-1 um-1
    """
    wavelength_range = np.linspace(wavelength_range_lower_limit.value, wavelength_range_upper_limit.value,
                                   100) * wavelength_range_upper_limit.unit
    blackbody_spectrum = BlackBody(temperature=temperature)(wavelength_range)

    units = blackbody_spectrum.unit
    blackbody_spectrum_binned = spectres.spectres(new_wavs=wavelength_bin_centers.to(u.um).value,
                                                  spec_wavs=wavelength_range.to(u.um).value,
                                                  spec_fluxes=blackbody_spectrum.value,
                                                  fill=0) * units
    return convert_blackbody_units(blackbody_spectrum_binned, wavelength_bin_centers, source_solid_angle)


def convert_blackbody_units(blackbody_spectrum_binned: np.ndarray,
                            wavelength_bin_centers: np.ndarray,
                            source_solid_angle: astropy.units.Quantity) -> np.ndarray:
    """Convert the binned black body spectrum from units erg / (Hz s sr cm2) to units ph / (m2 s um)

    :param blackbody_spectrum_binned: The binned blackbody spectrum
    :param wavelength_bin_centers: The wavelength bin centers
    :param source_solid_angle: The solid angle of the source
    :return: Array containing the spectral flux density in correct units
    """
    spectral_flux_density = np.zeros(len(blackbody_spectrum_binned)) * u.ph / u.m ** 2 / u.s / u.um

    for index in range(len(blackbody_spectrum_binned)):
        current_spectral_flux_density = (blackbody_spectrum_binned[index] * (source_solid_angle).to(u.sr)).to(
            u.ph / u.m ** 2 / u.s / u.um,
            equivalencies=u.spectral_density(
                wavelength_bin_centers[index]))
        spectral_flux_density[index] = current_spectral_flux_density
    return spectral_flux_density
