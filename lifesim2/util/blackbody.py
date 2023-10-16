import astropy.units
import numpy as np
from astropy import units as u
from astropy.modeling.models import BlackBody


def create_blackbody_spectrum(temperature,
                              wavelength_range_lower_limit: astropy.units.Quantity,
                              wavelength_range_upper_limit: astropy.units.Quantity,
                              wavelength_bin_centers: np.ndarray,
                              wavelength_bin_widths: np.ndarray,
                              angle_per_pixel: astropy.units.Quantity):
    wavelength_range = np.linspace(wavelength_range_lower_limit.value, wavelength_range_upper_limit.value,
                                   100) * wavelength_range_upper_limit.unit

    blackbody_spectrum = BlackBody(temperature=temperature)(wavelength_range)

    blackbody_spectrum_binned = bin_blackbody_spectrum(blackbody_spectrum, wavelength_range, wavelength_bin_centers,
                                                       wavelength_bin_widths)

    return convert_blackbody_to_flux(blackbody_spectrum_binned, angle_per_pixel, wavelength_bin_centers)


def bin_blackbody_spectrum(blackbody_spectrum, wavelength_range, wavelength_bin_centers, wavelength_bin_widths):
    blackbody_spectrum_binned = np.zeros(len(wavelength_bin_centers))
    units = blackbody_spectrum.unit
    for index, el in enumerate(wavelength_range):
        for index2, el_bin in enumerate(wavelength_bin_centers):
            bin_lower_edge = wavelength_bin_centers[index2] - wavelength_bin_widths[index2] / 2
            bin_upper_edge = wavelength_bin_centers[index2] + wavelength_bin_widths[index2] / 2
            if el >= bin_lower_edge and el < bin_upper_edge:
                blackbody_spectrum_binned[index2] += el.value
    return blackbody_spectrum_binned * units


def convert_blackbody_to_flux(blackbody_spectrum_binned, angle_per_pixel, wavelength_bin_centers):
    flux = np.zeros(len(blackbody_spectrum_binned)) * u.ph / u.m ** 2 / u.s / u.um

    for index in range(len(blackbody_spectrum_binned)):
        current_flux = (blackbody_spectrum_binned[index] * (angle_per_pixel[index] ** 2).to(u.sr)).to(
            u.ph / u.m ** 2 / u.s / u.um,
            equivalencies=u.spectral_density(
                wavelength_bin_centers[index]))
        # TODO: fix flux to high
        flux[index] = current_flux
    return flux
