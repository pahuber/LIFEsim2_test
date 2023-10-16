import astropy.units
import numpy as np

from astropy import units as u


class InstrumentParameters():
    def __init__(self,
                 aperture_diameter: astropy.units.Quantity,
                 wavelength_range_lower_limit: astropy.units.Quantity,
                 wavelength_range_upper_limit: astropy.units.Quantity,
                 spectral_resolving_power: astropy.units.Quantity):
        self.aperture_radius = aperture_diameter / 2
        self.wavelength_range_lower_limit = wavelength_range_lower_limit.to(u.um)
        self.wavelength_range_upper_limit = wavelength_range_upper_limit.to(u.um)
        self.spectral_resolving_power = spectral_resolving_power
        self.wavelength_bin_widths = None
        self.wavelength_bin_centers = None
        self._get_wavelength_bins()
        self.field_of_view = list(
            (wavelength.to(u.m) / self.aperture_radius.to(u.m)) * u.arcsec for wavelength in
            self.wavelength_bin_centers)

    def _get_wavelength_bins(self):
        """Return the wavelength bin widths. Implementation as in
        https://github.com/jlustigy/coronagraph/blob/master/coronagraph/noise_routines.py.

        :return:
        """
        minimum_bin_width = self.wavelength_range_lower_limit.value / self.spectral_resolving_power.value
        maximum_bin_width = self.wavelength_range_upper_limit.value / self.spectral_resolving_power.value
        wavelength = self.wavelength_range_lower_limit.value

        number_of_wavelengths = 1

        while (wavelength < self.wavelength_range_upper_limit.value + maximum_bin_width):
            wavelength = wavelength + wavelength / self.spectral_resolving_power.value
            number_of_wavelengths = number_of_wavelengths + 1

        wavelengths = np.zeros(number_of_wavelengths)
        wavelengths[0] = self.wavelength_range_lower_limit.value

        for index in range(1, number_of_wavelengths):
            wavelengths[index] = wavelengths[index - 1] + wavelengths[
                index - 1] / self.spectral_resolving_power.value
        number_of_wavelengths = len(wavelengths)
        wavelength_bins = np.zeros(number_of_wavelengths)

        # Set wavelength widths
        for index in range(1, number_of_wavelengths - 1):
            wavelength_bins[index] = 0.5 * (wavelengths[index + 1] + wavelengths[index]) - 0.5 * (
                    wavelengths[index - 1] + wavelengths[index])

        # Set edges to be same as neighbor
        wavelength_bins[0] = minimum_bin_width
        wavelength_bins[number_of_wavelengths - 1] = maximum_bin_width

        wavelengths = wavelengths[:-1]
        wavelength_bins = wavelength_bins[:-1]

        self.wavelength_bin_centers = wavelengths * u.um
        self.wavelength_bin_widths = wavelength_bins * u.um
