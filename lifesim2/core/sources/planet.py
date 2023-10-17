import astropy.units
import numpy as np
from astropy import units as u

from lifesim2.core.sources.source import Source


class Planet(Source, object):
    def __init__(self,
                 name: str,
                 radius: astropy.units.Quantity,
                 mass: astropy.units.Quantity,
                 temperature: astropy.units.Quantity,
                 star_separation: astropy.units.Quantity,
                 star_distance: astropy.units.Quantity,
                 number_of_wavelength_bins: int) -> object:
        super().__init__(number_of_wavelength_bins=number_of_wavelength_bins)
        self.name = name
        self.radius = radius
        self.mass = mass
        self.temperature = temperature
        self.star_separation = star_separation
        self.star_distance = star_distance
        self.star_angular_separation = (self.star_separation / self.star_distance * u.rad).to(u.arcsec)
        self.solid_angle = np.pi * (((self.radius.to(u.m)) ** 2 / (self.star_distance.to(u.m))) * u.rad) ** 2

        @property
        def position(self) -> astropy.units.Quantity:
            """Return the (x, y) position in arcseconds.

            :return: A tuple containing the x- and y-position.
            """
            # TODO: implement planet position correctly
            x = self.star_angular_separation * np.cos(np.pi / 4)
            y = self.star_angular_separation * np.sin(np.pi / 4)
            return (x, y)
