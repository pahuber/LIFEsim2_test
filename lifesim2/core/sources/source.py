from abc import ABC
from typing import Any

import astropy
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
