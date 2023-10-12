import astropy.units
import numpy as np

from lifesim2.core.observation.sources.source import Source
from lifesim2.util.blackbody import get_blackbody_spectrum


class Star(Source):
    def __init__(self,
                 label: str,
                 radius: astropy.units.Quantity,
                 mass: astropy.units.Quantity,
                 temperature: astropy.units.Quantity,
                 distance: astropy.units.Quantity):
        self.label = label
        self.radius = radius
        self.mass = mass
        self.temperature = temperature
        self.distance = distance

    def create_blackbody_spectrum(self, wavelength_range: np.ndarray):
        self.spectrum = get_blackbody_spectrum(temperature=self.temperature, wavelength_range=wavelength_range)
