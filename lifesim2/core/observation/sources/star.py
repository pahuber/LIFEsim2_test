import astropy.units

from lifesim2.core.observation.sources.source import Source


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
