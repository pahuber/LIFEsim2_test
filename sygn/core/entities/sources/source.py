from abc import ABC, abstractmethod
from typing import Any, Tuple

import astropy
import numpy as np
from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from sygn.io.validators import validate_quantity_units


class Source(ABC, BaseModel):
    """Class representation of a photon source.
    """

    # number_of_wavelength_bins: int
    name: str
    temperature: Any

    mean_spectral_flux_density: Any = None

    @field_validator('temperature')
    def _validate_temperature(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the temperature input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The temperature in units of temperature
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.K)

    @abstractmethod
    def get_sky_coordinate_maps(self, time: astropy.units.Quantity) -> Tuple:
        """Return the sky coordinate maps of the source. The intensity responses are calculated in a resolution that
        allows the source to fill the grid, thus, each source needs to define its own sky coordinate map.

        :param time: The time
        :return: A tuple containing the x- and y-sky coordinate maps
        """
        pass

    @abstractmethod
    def get_sky_position_map(self, time: astropy.units.Quantity) -> np.ndarray:
        """Return the sky position map of the source object. Consists of zeros everywhere, but at the source position, where
        it is one.

        :param time: The time
        :return: The sky position map
        """
        pass
