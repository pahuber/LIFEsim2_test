from abc import ABC, abstractmethod
from typing import Any

import astropy
import numpy as np
from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from lifesim2.io.validators import validate_quantity_units


class Source(ABC, BaseModel):
    """Class representation of a photon source.
    """

    # number_of_wavelength_bins: int
    name: str
    temperature: Any

    flux: Any = None

    @field_validator('temperature')
    def validate_temperature(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the temperature input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The temperature in units of temperature
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.K)

    @property
    def shape_map(self):
        """Return the shape map of the source. Consists of a 2D map that is zero everywhere, but at the pixels where the
        source is present, where it is one.

        :return: The shape map
        """
        return self.get_shape_map()

    @property
    def sky_coordinate_maps(self):
        """Return the sky coordinate maps of the source. The intensity responses are calculated in a resolution that
        allows the source to fill the grid, thus, each source needs to define its own sky coordinate map.

        :return: The sky coordinate maps
        """
        return self.get_sky_coordinate_maps()

    @abstractmethod
    def get_sky_coordinate_maps(self) -> np.ndarray:
        """Return the x- and y-sky-coordinate maps. Extend the coordinates by 10% of the planet star separation.

        :return: A tuple containing the x- and y-sky coordinate maps
        """
        pass

    @abstractmethod
    def get_shape_map(self) -> np.ndarray:
        """Return the shape map of the planets. Consists of zero everywhere, but at the planet position, where it is one.

        :return: The shape map
        """
        pass
