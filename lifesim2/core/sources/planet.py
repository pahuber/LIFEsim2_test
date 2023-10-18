from typing import Any

import astropy
import numpy as np
from astropy import units as u
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo

from lifesim2.core.sources.source import Source
from lifesim2.io.validators import validate_quantity_units


class Planet(Source):
    name: str
    temperature: Any
    radius: Any
    mass: Any
    star_separation: Any
    star_distance: Any
    number_of_wavelength_bins: int

    @field_validator('radius')
    def validate_radius(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)

    @field_validator('mass')
    def validate_mass(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.kg)

    @field_validator('star_separation')
    def validate_star_separation(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)

    @field_validator('star_distance')
    def validate_star_distance(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.m)

    @property
    def solid_angle(self):
        return np.pi * (((self.radius.to(u.m)) ** 2 / (self.distance.to(u.m))) * u.rad) ** 2

    @property
    def star_angular_separation(self):
        return (self.star_separation / self.star_distance * u.rad).to(u.arcsec)

    @property
    def position(self) -> astropy.units.Quantity:
        """Return the (x, y) position in arcseconds.

        :return: A tuple containing the x- and y-position.
        """
        # TODO: implement planet position correctly
        x = self.star_angular_separation * np.cos(0)
        y = self.star_angular_separation * np.sin(0)
        return (x, y)
