from typing import Any

import numpy as np
from astropy import units as u
from tqdm import tqdm

from sygn.core.context import Context
from sygn.core.entities.photon_sources.photon_source import PhotonSource
from sygn.util.blackbody import create_blackbody_spectrum
from sygn.util.grid import get_meshgrid, get_radial_map
from sygn.util.helpers import Coordinates


class Exozodi(PhotonSource):
    """Class representation of an exozodi.
    """
    level: float
    inclination: Any
    star_distance: Any
    star_luminosity: Any
    field_of_view_in_au_radial_maps: Any = None

    def _calculate_sky_coordinates(self, context: Context) -> np.ndarray:
        sky_coordinates = np.zeros(len(context.observatory.instrument_parameters.field_of_view), dtype=object)

        # The sky coordinates have a different extent for each field of view, i.e. for each wavelength
        for index_fov in range(len(context.observatory.instrument_parameters.field_of_view)):
            sky_coordinates_at_fov = get_meshgrid(
                context.observatory.instrument_parameters.field_of_view[index_fov].to(u.rad),
                context.settings.grid_size)
            sky_coordinates[index_fov] = Coordinates(sky_coordinates_at_fov[0], sky_coordinates_at_fov[1])
        return sky_coordinates

    def _calculate_sky_brightness_distribution(self, context: Context) -> np.ndarray:
        # The sky brightness distribution is calculated as in Dannert et al., 2022
        reference_radius = np.sqrt(self.star_luminosity.to(u.Lsun)).value * u.au
        surface_maps = self.level * 7.12e-8 * (
                self.field_of_view_in_au_radial_maps / reference_radius) ** (-0.34)
        return surface_maps * self.mean_spectral_flux_density

    def _calculate_mean_spectral_flux_density(self, context: Context) -> np.ndarray:
        # TODO: Fix calculation
        # Calculate the fields of view in AU from the angular fields of view
        fields_of_view_in_au = (
                context.observatory.instrument_parameters.field_of_view / u.rad * self.star_distance).to(u.au)

        # Initialize empty maps
        temperature_map = np.zeros(
            (len(fields_of_view_in_au), context.settings.grid_size, context.settings.grid_size)) * u.K
        self.field_of_view_in_au_radial_maps = np.zeros(
            (
                len(fields_of_view_in_au), context.settings.grid_size,
                context.settings.grid_size)) * u.au
        mean_spectral_flux_density = np.zeros(temperature_map.shape) * u.ph / (u.m ** 2 * u.um * u.s)

        # For each field of view, i.e. each wavelength, calculate the temperature profile and the mean spectral flux
        # density
        for index_fov, field_of_view in enumerate(tqdm(fields_of_view_in_au)):
            self.field_of_view_in_au_radial_maps[index_fov] = get_radial_map(field_of_view,
                                                                             context.settings.grid_size)
            temperature_map[index_fov] = self._get_exozodi_temperature(
                self.field_of_view_in_au_radial_maps[index_fov])

            for index_x in range(context.settings.grid_size):
                for index_y in range(context.settings.grid_size):
                    mean_spectral_flux_density[index_fov][index_x][index_y] = (create_blackbody_spectrum(
                        temperature_map[index_fov][index_x][index_y],
                        context.observatory.instrument_parameters.wavelength_range_lower_limit,
                        context.observatory.instrument_parameters.wavelength_range_upper_limit,
                        context.observatory.instrument_parameters.wavelength_bin_centers,
                        context.observatory.instrument_parameters.wavelength_bin_widths,
                        context.observatory.instrument_parameters.field_of_view ** 2
                    )[index_fov])
        return mean_spectral_flux_density.reshape((-1, context.settings.grid_size, context.settings.grid_size))

    def _get_exozodi_temperature(self, maximum_stellar_separations_radial_map) -> np.ndarray:
        """Return a 2D map corresponding to the temperature distribution of the exozodi.

        :param maximum_stellar_separations_radial_map: The 2D map corresponding to the maximum radial stellar
        separations
        :return: The temperature distribution map
        """
        return (278.3 * self.star_luminosity.to(u.Lsun) ** 0.25 * maximum_stellar_separations_radial_map ** (
            -0.5)).value * u.K

    def get_sky_coordinates(self, index_time: int, index_wavelength: int) -> Coordinates:
        return self.sky_coordinates[index_wavelength]

    def get_sky_brightness_distribution(self, index_time: int, index_wavelength: int) -> np.ndarray:
        return self.sky_brightness_distribution[index_wavelength]
