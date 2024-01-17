from typing import Any, Optional

from pydantic import BaseModel

from sygn.core.entities.noise import Noise


class Settings(BaseModel):
    """Class representation of the simulation configurations.

    """
    grid_size: int
    time_steps: int
    planet_orbital_motion: bool
    noise: Optional[Noise]
    integration_time: Any = None
    time_step: Any = None

    def __init__(self, **data):
        """Constructor method.

        :param data: Data to initialize the star class.
        """
        super().__init__(**data)
        self.time_step = self.integration_time / self.time_steps
        self.noise.get_phase_perturbations_distribution(self.time_step)
