from abc import ABC

import numpy as np
from astropy import units as u


class Source(ABC):

    def __init__(self, number_of_wavelength_bins: int):
        self.name = None
        self.number_of_wavelength_bins = number_of_wavelength_bins
        self.flux = np.zeros(self.number_of_wavelength_bins) * u.ph / u.m ** 2 / u.s / u.um
        self.temperature = None
