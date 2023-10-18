from typing import Any, Tuple

import astropy
import numpy as np
from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from lifesim2.io.validators import validate_quantity_units


class InstrumentParameters(BaseModel):
    aperture_diameter: Any
    wavelength_range_lower_limit: Any
    wavelength_range_upper_limit: Any
    spectral_resolving_power: int
    field_of_view: Any = None

    @field_validator('aperture_diameter')
    def validate_aperture_diameter(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the aperture diameter input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The aperture diameter in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)

    @field_validator('wavelength_range_lower_limit')
    def validate_wavelength_range_lower_limit(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the wavelength range lower limit input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The lower wavelength range limit in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)

    @field_validator('wavelength_range_upper_limit')
    def validate_wavelength_range_upper_limit(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the wavelength range upper limit input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The upper wavelength range limit in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)

    @property
    def aperture_radius(self) -> astropy.unit.Quantity:
        """Return the aperture radius.

        :return: The aperture radius
        """
        return self.aperture_diameter / 2

    @property
    def wavelength_bin_centers(self) -> np.ndarray:
        """Return the wavelength bin centers.

        :return: An array containing the wavelength bin centers
        """
        return self._get_wavelength_bins()[0]

    @property
    def wavelength_bin_widths(self) -> np.ndarray:
        """Return the wavelength bin widths.

        :return: An array containing the wavelength bin widths
        """
        return self._get_wavelength_bins()[1]

    def _get_wavelength_bins(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the wavelength bin centers and widths. Implementation as in
        https://github.com/jlustigy/coronagraph/blob/master/coronagraph/noise_routines.py.

        :return: A tuple containing the wavelength bin centers and widths
        """
        # TODO: clean up
        minimum_bin_width = self.wavelength_range_lower_limit.value / self.spectral_resolving_power
        maximum_bin_width = self.wavelength_range_upper_limit.value / self.spectral_resolving_power
        wavelength = self.wavelength_range_lower_limit.value

        number_of_wavelengths = 1

        while (wavelength < self.wavelength_range_upper_limit.value + maximum_bin_width):
            wavelength = wavelength + wavelength / self.spectral_resolving_power
            number_of_wavelengths = number_of_wavelengths + 1

        wavelengths = np.zeros(number_of_wavelengths)
        wavelengths[0] = self.wavelength_range_lower_limit.value

        for index in range(1, number_of_wavelengths):
            wavelengths[index] = wavelengths[index - 1] + wavelengths[
                index - 1] / self.spectral_resolving_power
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

        return wavelengths * u.um, wavelength_bins * u.um
