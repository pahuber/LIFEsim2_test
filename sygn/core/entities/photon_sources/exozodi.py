from typing import Any

import astropy
import numpy as np
from astropy import units as u

from sygn.core.entities.photon_sources.photon_source import PhotonSource
from sygn.util.grid import get_meshgrid
from sygn.util.helpers import Coordinates


class Exozodi(PhotonSource):
    """Class representation of a exozodi.
    """
    level: float
    inclination: Any
    reference_radius: Any
    maximum_stellar_separations_radial_maps: Any

    def get_sky_coordinates(self, time: astropy.units.Quantity, grid_size: int,
                            fields_of_view: astropy.units.Quantity = None) -> Coordinates:
        all_sky_coordinates = []
        # self.grid_size = grid_size
        for index_fov in range(len(fields_of_view)):
            sky_coordinates = get_meshgrid(fields_of_view[index_fov].to(u.rad), grid_size)
            all_sky_coordinates.append(Coordinates(sky_coordinates[0], sky_coordinates[1]))
        return all_sky_coordinates

    def get_sky_brightness_distribution_map(self, sky_coordinates: Coordinates) -> np.ndarray:
        surface_maps = self.level * 7.12e-8 * (
                self.maximum_stellar_separations_radial_maps / self.reference_radius) ** (-0.34)
        return surface_maps * self.mean_spectral_flux_density
