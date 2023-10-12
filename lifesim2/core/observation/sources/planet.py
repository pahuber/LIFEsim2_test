import astropy.units

from lifesim2.core.observation.sources.source import Source


class Planet(Source):
    def __init__(self,
                 label: str,
                 radius: astropy.units.Quantity,
                 mass: astropy.units.Quantity,
                 temperature: astropy.units.Quantity,
                 star_separation: astropy.units.Quantity):
        self.label = label
        self.radius = radius
        self.mass = mass
        self.temperature = temperature
        self.star_separation = star_separation
        self.spectrum = None

    def create_blackbody_spectrum(self):
        # self.spectrum = get_blackbody_spectrum()
        pass
