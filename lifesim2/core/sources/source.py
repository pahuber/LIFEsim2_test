from abc import ABC

import numpy as np
from astropy import units as u


class Source(ABC):
    number_of_steps = 1000

    def __init__(self):
        self.name = None
        self.flux = np.zeros(Source.number_of_steps) * u.ph / u.m ** 2 / u.s / u.um
        self.temperature = None
