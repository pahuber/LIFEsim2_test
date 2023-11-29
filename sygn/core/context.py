import numpy as np
from astropy import units as u


class Context():
    """Class representation of the contexts.
    """

    def __init__(self):
        """Constructor method.
        """
        self.settings = None
        self.mission = None
        self.observatory = None
        self.target_specific_photon_sources = []
        self.target_unspecific_photon_sources = []
        self.data = None
        self.templates = []
        self.animator = None
        self.star_habitable_zone_central_angular_radius = None

    @property
    def time_range(self) -> np.ndarray:
        """Return the time range.

        :return: The time range
        """
        return np.arange(0, self.mission.integration_time.to(u.s).value,
                         self.settings.time_step.to(u.s).value) * u.s
