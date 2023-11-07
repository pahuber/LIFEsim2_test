from typing import Any, Optional

import astropy
from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from sygn.core.modules.settings.noise_contributions import NoiseContributions
from sygn.io.validators import validate_quantity_units


class Settings(BaseModel):
    """Class representation of the simulation configurations.

    """
    grid_size: int
    time_step: Any
    planet_orbital_motion: bool
    noise_contributions: Optional[NoiseContributions]

    def __init__(self, **data):
        """Constructor method.

        :param data: Data to initialize the star class.
        """
        super().__init__(**data)
        self.noise_contributions.get_optical_path_difference_distribution(self.time_step)

    @field_validator('time_step')
    def _validate_time_step(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the time step input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The time step in units of time
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.s)
