from typing import Any, Optional

import astropy
from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from lifesim2.core.noise_contributions import NoiseContributions
from lifesim2.io.validators import validate_quantity_units


class SimulationConfiguration(BaseModel):
    """Class representation of the simulation configurations.

    """
    grid_size: int
    time_step: Any
    noise_contributions: Optional[NoiseContributions]
    time_range: Any = None

    @field_validator('time_step')
    def validate_time_step(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the time step input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The time step in units of time
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.s)
