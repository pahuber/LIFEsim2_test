import numpy as np
from astropy import units as u

from sygn.core.contexts.base_context import BaseContext


class GeneratorContext(BaseContext):
    """Class representation of the generator contexts.
    """

    def __init__(self):
        """Constructor method.
        """
        self.settings = None
        self.observation = None
        self.observatory = None
        self.photon_sources = []
        self.data = []
        self.animator = None

    @property
    def time_range(self) -> np.ndarray:
        """Return the time range.

        :return: The time range
        """
        return np.arange(0, self.observation.integration_time.to(u.s).value,
                         self.settings.time_step.to(u.s).value) * u.s
