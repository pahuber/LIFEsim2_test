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

    number_of_wavelength_bins: int
    name: str
    temperature: Any
    solid_angle: Any = None
    flux: Any = None

    @field_validator('temperature')
    def validate_temperature(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.K)

    @property
    def shape_map(self):
        return self.get_shape_map()

    @property
    def sky_coordinate_maps(self):
        return self.get_sky_coordinate_maps()

    @abstractmethod
    def get_sky_coordinate_maps(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_shape_map(self) -> np.ndarray:
        pass
