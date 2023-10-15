import astropy.units

from lifesim2.core.sources.source import Source


class Star(Source):
    def __init__(self,
                 name: str,
                 radius: astropy.units.Quantity,
                 mass: astropy.units.Quantity,
                 temperature: astropy.units.Quantity,
                 distance: astropy.units.Quantity):
        super().__init__()
        self.name = name
        self.radius = radius
        self.mass = mass
        self.temperature = temperature
        self.distance = distance
