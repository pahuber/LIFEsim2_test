from typing import Any, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic

from sygn.core.context import Context
from sygn.core.entities.photon_sources.photon_source import PhotonSource
from sygn.util.blackbody import create_blackbody_spectrum
from sygn.util.grid import get_meshgrid
from sygn.util.helpers import Coordinates


class LocalZodi(PhotonSource):
    """Class representation of a local zodi.
    """

    star_right_ascension: Any
    star_declination: Any

    def _get_ecliptic_coordinates(self) -> Tuple:
        """Return the ecliptic latitude and relative ecliptic longitude.

        :return: Tuple containing the two coordinates
        """
        coordinates = SkyCoord(ra=self.star_right_ascension, dec=self.star_declination, frame='icrs')
        coordinates_ecliptic = coordinates.transform_to(GeocentricTrueEcliptic)
        ecliptic_latitude = coordinates_ecliptic.lat.to(u.deg)
        ecliptic_longitude = coordinates_ecliptic.lon.to(u.deg)
        relative_ecliptic_longitude = ecliptic_longitude - 0 * u.deg
        return ecliptic_latitude, relative_ecliptic_longitude

    def _calculate_sky_coordinates(self, context: Context) -> Coordinates:
        sky_coordinates = np.zeros(len(context.observatory.instrument_parameters.fields_of_view), dtype=object)
        for index_fov in range(len(context.observatory.instrument_parameters.fields_of_view)):
            sky_coordinates_at_fov = get_meshgrid(
                context.observatory.instrument_parameters.fields_of_view[index_fov].to(u.rad),
                context.settings.grid_size)
            sky_coordinates[index_fov] = Coordinates(sky_coordinates_at_fov[0], sky_coordinates_at_fov[1])
        return sky_coordinates

    def _calculate_sky_brightness_distribution(self, context: Context) -> np.ndarray:
        grid = np.ones((context.settings.grid_size, context.settings.grid_size))
        return np.einsum('i, jk ->ijk', self.mean_spectral_flux_density, grid)

    def _calculate_mean_spectral_flux_density(self, context: Context) -> np.ndarray:
        variable_tau = 4e-8
        variable_a = 0.22
        ecliptic_latitude, relative_ecliptic_longitude = self._get_ecliptic_coordinates()
        mean_spectral_flux_density = (
                variable_tau
                * (create_blackbody_spectrum(265 * u.K,
                                             context.observatory.instrument_parameters.wavelength_range_lower_limit,
                                             context.observatory.instrument_parameters.wavelength_range_upper_limit,
                                             context.observatory.instrument_parameters.wavelength_bin_centers,
                                             context.observatory.instrument_parameters.wavelength_bin_widths,
                                             context.observatory.instrument_parameters.fields_of_view ** 2)
                   + variable_a
                   * create_blackbody_spectrum(5778 * u.K,
                                               context.observatory.instrument_parameters.wavelength_range_lower_limit,
                                               context.observatory.instrument_parameters.wavelength_range_upper_limit,
                                               context.observatory.instrument_parameters.wavelength_bin_centers,
                                               context.observatory.instrument_parameters.wavelength_bin_widths,
                                               context.observatory.instrument_parameters.fields_of_view ** 2)
                   * ((1 * u.Rsun).to(u.au) / (1.5 * u.au)) ** 2)
                * (
                        (np.pi / np.arccos(
                            np.cos(relative_ecliptic_longitude) * np.cos(ecliptic_latitude))) / (
                                np.sin(ecliptic_latitude) ** 2 + 0.6 * (
                                context.observatory.instrument_parameters.wavelength_bin_centers / (
                                11 * u.um)) ** (
                                    -0.4) * np.cos(ecliptic_latitude) ** 2)) ** 0.5)
        return mean_spectral_flux_density

    def get_sky_coordinates(self, index_time: int, index_wavelength: int) -> Coordinates:
        return self.sky_coordinates[index_wavelength]

    def get_sky_brightness_distribution(self, index_time: int, index_wavelength: int) -> np.ndarray:
        return self.sky_brightness_distribution[index_wavelength]
