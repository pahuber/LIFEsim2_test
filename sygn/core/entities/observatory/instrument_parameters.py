from typing import Any, Tuple

import astropy
import numpy as np
from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from sygn.io.validators import validate_quantity_units


class InstrumentParameters(BaseModel):
    """Class representation of instrument parameters.
    """
    aperture_diameter: Any
    spectral_resolving_power: int
    wavelength_range_lower_limit: Any
    wavelength_range_upper_limit: Any
    unperturbed_instrument_throughput: float

    @field_validator('aperture_diameter')
    def _validate_aperture_diameter(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the aperture diameter input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The aperture diameter in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m).to(u.m)

    @field_validator('wavelength_range_lower_limit')
    def _validate_wavelength_range_lower_limit(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the wavelength range lower limit input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The lower wavelength range limit in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m).to(u.um)

    @field_validator('wavelength_range_upper_limit')
    def _validate_wavelength_range_upper_limit(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the wavelength range upper limit input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The upper wavelength range limit in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m).to(u.um)

    @property
    def aperture_radius(self) -> astropy.units.Quantity:
        """Return the aperture radius.

        :return: The aperture radius
        """
        return self.aperture_diameter / 2

    @property
    def field_of_view(self) -> np.ndarray:
        """Return the fields of view for each wavelength.

        :return: An array containing the field of view for each wavelength
        """
        return (self.wavelength_bin_centers.to(u.m) / self.aperture_diameter * u.rad)

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
        """Return the wavelength bin centers and widths. The wavelength bin widths are calculated starting from the
        wavelength lower range. As a consequence, the uppermost wavelength bin might be smaller than anticipated.

        :return: A tuple containing the wavelength bin centers and widths
        """
        current_minimum_wavelength = self.wavelength_range_lower_limit.value
        wavelength_bin_centers = []
        wavelength_bin_widths = []

        while current_minimum_wavelength <= self.wavelength_range_upper_limit.value:
            center_wavelength = current_minimum_wavelength / (1 - 1 / (2 * self.spectral_resolving_power))
            bin_width = 2 * (center_wavelength - current_minimum_wavelength)
            if center_wavelength + bin_width / 2 <= self.wavelength_range_upper_limit.value:
                wavelength_bin_centers.append(center_wavelength)
                wavelength_bin_widths.append(bin_width)
                current_minimum_wavelength = center_wavelength + bin_width / 2
            else:
                last_bin_width = self.wavelength_range_upper_limit.value - current_minimum_wavelength
                last_center_wavelength = self.wavelength_range_upper_limit.value - last_bin_width / 2
                wavelength_bin_centers.append(last_center_wavelength)
                wavelength_bin_widths.append(last_bin_width)
                break
        return (np.array(wavelength_bin_centers) * u.um, np.array(wavelength_bin_widths) * u.um)
