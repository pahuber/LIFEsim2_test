import numpy as np
from astropy import units as u


class Context():
    def __init__(self):
        self.settings = None
        self.observation = None
        self.observatory = None
        self.target_systems = []
        self.differential_photon_counts = None

    @property
    def time_range(self):
        return np.arange(0, self.observation.integration_time.to(u.s).value,
                         self.settings.time_step.to(u.s).value) * u.s
