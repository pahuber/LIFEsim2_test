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

    @property
    def time_range_planet_motion(self) -> np.ndarray:
        """Return the time range considering whether the planet orbital motion is modeled. This is identical to the
        time_range property, if orbital motion is modeled, otherwise it is a range of constant values corresponding to
        the initial time.

        :return: The time range
        """
        if self.settings.planet_orbital_motion:
            return self.time_range
        else:
            return [self.time_range[0]]
