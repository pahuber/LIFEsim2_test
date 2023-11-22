from typing import Any

import astropy
import numpy as np
from astropy import units as u

from sygn.core.entities.photon_sources.photon_source import PhotonSource
from sygn.util.grid import get_meshgrid
from sygn.util.helpers import Coordinates


class LocalZodi(PhotonSource):
    """Class representation of a local zodi.
    """

    # star_right_ascension: Any
    # star_declination: Any
    fields_of_view: Any = None
    grid_size: int = None

    def get_sky_coordinates(self, time: astropy.units.Quantity, grid_size: int,
                            fields_of_view: astropy.units.Quantity = None) -> Coordinates:
        all_sky_coordinates = []
        self.grid_size = grid_size
        for index_fov in range(len(fields_of_view)):
            sky_coordinates = get_meshgrid(fields_of_view[index_fov].to(u.rad), grid_size)
            all_sky_coordinates.append(Coordinates(sky_coordinates[0], sky_coordinates[1]))
        return all_sky_coordinates

    def get_sky_brightness_distribution_map(self, sky_coordinates: Coordinates) -> np.ndarray:
        map = np.ones((self.grid_size, self.grid_size))
        position_map = np.einsum('i, jk ->ijk', self.mean_spectral_flux_density, map)
        # position_map = np.ones(((len(self.mean_spectral_flux_density),) + sky_coordinates.x.shape)) * \
        #                self.mean_spectral_flux_density[0].unit
        # radius_map = (np.sqrt(sky_coordinates[0] ** 2 + sky_coordinates[
        #     1] ** 2) <= self.angular_radius)
        #
        # for index_wavelength in range(len(self.mean_spectral_flux_density)):
        #     position_map[index_wavelength] = radius_map * self.mean_spectral_flux_density[index_wavelength]

        return position_map
