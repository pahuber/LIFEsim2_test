import astropy.units
from astropy import units as u

from lifesim2.core.observation.sources.source import Source


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
