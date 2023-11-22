import astropy
import numpy as np

from sygn.core.entities.photon_sources.photon_source import PhotonSource
from sygn.util.grid import get_meshgrid
from sygn.util.helpers import Coordinates


class Exozodi(PhotonSource):
    """Class representation of a exozodi.
    """
    level: float

    def get_sky_coordinates(self, time: astropy.units.Quantity) -> Coordinates:
        """Return the sky coordinate maps of the source. The intensity responses are calculated in a resolution that
        allows the source to fill the grid, thus, each source needs to define its own sky coordinate map. Add 10% to the
        angular radius to account for rounding issues and make sure the source is fully covered within the map.

        :param time: The time
        :return: A tuple containing the x- and y-sky coordinate maps
        """
        return get_meshgrid(2 * (1.05 * self.angular_radius), self.grid_size)

    def get_sky_brightness_distribution_map(self, time: astropy.units.Quantity) -> np.ndarray:
        sky_coordinate_maps = self.get_sky_coordinates(time)
        position_map = np.zeros(((len(self.mean_spectral_flux_density),) + sky_coordinate_maps[0].shape)) * \
                       self.mean_spectral_flux_density[0].unit
        radius_map = (np.sqrt(sky_coordinate_maps[0] ** 2 + sky_coordinate_maps[
            1] ** 2) <= self.angular_radius)

        for index_wavelength in range(len(self.mean_spectral_flux_density)):
            position_map[index_wavelength] = radius_map * self.mean_spectral_flux_density[index_wavelength]

        return position_map
