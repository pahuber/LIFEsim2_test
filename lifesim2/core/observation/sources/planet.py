import astropy.units
import numpy as np
from astropy import units as u

from lifesim2.core.observation.sources.source import Source
from lifesim2.util.blackbody import get_blackbody_spectrum


class Planet(Source):
    def __init__(self,
                 label: str,
                 radius: astropy.units.Quantity,
                 mass: astropy.units.Quantity,
                 temperature: astropy.units.Quantity,
                 star_separation: astropy.units.Quantity,
                 star_distance: astropy.units.Quantity):
        self.label = label
        self.radius = radius
        self.mass = mass
        self.temperature = temperature
        self.star_separation = star_separation
        self.star_distance = star_distance
        self.star_angular_separation = (self.star_separation / self.star_distance * u.rad).to(u.arcsec)
        self.spectrum = None

    def create_blackbody_spectrum(self, wavelength_range: np.ndarray):
        self.spectrum = get_blackbody_spectrum(temperature=self.temperature, wavelength_range=wavelength_range)
