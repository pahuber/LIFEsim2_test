from abc import ABC

import numpy as np
from astropy import units as u


class Source(ABC, object):
    """Class representation of a photon source.
    """

    def __init__(self, number_of_wavelength_bins: int):
        """Constructor method.

        :param number_of_wavelength_bins: The number of wavelength bins used in the observation
        """
        self.number_of_wavelength_bins = number_of_wavelength_bins
        self.name = None
        self.temperature = None
        self.position = None
        self.solid_angle = None
        self.flux = np.zeros(self.number_of_wavelength_bins) * u.ph / u.m ** 2 / u.s / u.um
