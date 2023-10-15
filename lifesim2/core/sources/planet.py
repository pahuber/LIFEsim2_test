import astropy.units
from astropy import units as u

from lifesim2.core.sources.source import Source


class Planet(Source):
    def __init__(self,
                 name: str,
                 radius: astropy.units.Quantity,
                 mass: astropy.units.Quantity,
                 temperature: astropy.units.Quantity,
                 star_separation: astropy.units.Quantity,
                 star_distance: astropy.units.Quantity) -> object:
        super().__init__()
        self.name = name
        self.radius = radius
        self.mass = mass
        self.temperature = temperature
        self.star_separation = star_separation
        self.star_distance = star_distance
        self.star_angular_separation = (self.star_separation / self.star_distance * u.rad).to(u.arcsec)
