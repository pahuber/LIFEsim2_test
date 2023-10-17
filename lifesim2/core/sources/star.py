import astropy.units
import numpy as np
from astropy import units as u

from lifesim2.core.sources.source import Source


class Star(Source):
    def __init__(self,
                 name: str,
                 radius: astropy.units.Quantity,
                 mass: astropy.units.Quantity,
                 temperature: astropy.units.Quantity,
                 distance: astropy.units.Quantity,
                 number_of_wavelength_bins: int):
        super().__init__(number_of_wavelength_bins=number_of_wavelength_bins)
        self.name = name
        self.radius = radius
        self.mass = mass
        self.temperature = temperature
        self.distance = distance
        self.solid_angle = np.pi * (((self.radius.to(u.m)) ** 2 / (self.distance.to(u.m))) * u.rad) ** 2
        self.position = (0, 0) * u.arcsec
