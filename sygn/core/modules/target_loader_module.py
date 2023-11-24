from pathlib import Path
from typing import Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic
from tqdm import tqdm

from sygn.core.context import Context
from sygn.core.entities.photon_sources.exozodi import Exozodi
from sygn.core.entities.photon_sources.local_zodi import LocalZodi
from sygn.core.entities.photon_sources.planet import Planet
from sygn.core.entities.photon_sources.star import Star
from sygn.io.config_reader import ConfigReader
from sygn.util.blackbody import create_blackbody_spectrum
from sygn.util.grid import get_radial_map


class TargetLoaderModule():
    """Class representation of the target loader module.
    """

    def __init__(self,
                 path_to_context_file: Path,
                 path_to_spectrum_file: Path = None):
        """Constructor method.

        :param path_to_context_file: Path to the context file
        :param path_to_spectrum_file: Path to the spectrum file
        """
        self._path_to_context_file = path_to_context_file
        self._path_to_spectrum_file = path_to_spectrum_file
        self.star = None
        self.planets = None
        self.exozodi = None
        self.local_zodi = None

    def _add_target_specific_photon_sources(self, context) -> list:
        """Add the photon sources to the list, if specified in the configurations.

        :param context: The context
        :return: The list containing the specified photon sources
        """
        target_specific_photon_sources = []
        if context.settings.noise_contributions.stellar_leakage:
            target_specific_photon_sources.append(self.star)
        if context.settings.noise_contributions.local_zodi_leakage:
            target_specific_photon_sources.append(
                self.local_zodi)
        if context.settings.noise_contributions.exozodi_leakage:
            target_specific_photon_sources.append(self.exozodi)
        for planet in self.planets:
            target_specific_photon_sources.append(planet)
        return target_specific_photon_sources

    def _get_ecliptic_coordinates(self) -> Tuple:
        """Return the ecliptic latitude and relative ecliptic longitude.

        :return: Tuple containing the two coordinates
        """
        coordinates = SkyCoord(ra=self.star.right_ascension, dec=self.star.declination, frame='icrs')
        coordinates_ecliptic = coordinates.transform_to(GeocentricTrueEcliptic)
        ecliptic_latitude = coordinates_ecliptic.lat.to(u.deg)
        ecliptic_longitude = coordinates_ecliptic.lon.to(u.deg)
        relative_ecliptic_longitude = ecliptic_longitude - 0 * u.deg
        return ecliptic_latitude, relative_ecliptic_longitude

    def _get_exozodi_temperature(self, maximum_stellar_separations_radial_map) -> np.ndarray:
        """Return a 2D map corresponding to the temperature distribution of the exozodi.

        :param maximum_stellar_separations_radial_map: The 2D map corresponding to the maximum radial stellar
        separations
        :return: The temperature distribution map
        """
        return (278.3 * self.star.luminosity.to(u.Lsun) ** 0.25 * maximum_stellar_separations_radial_map ** (
            -0.5)).value * u.K

    def _load_exozodi(self, config_dict: dict, context: Context) -> Exozodi:
        """Return the exozodi object from the dictionary.

        :param config_dict: The dictionary
        :param context: The context
        :return: The exozodi object
        """

        maximum_stellar_separations_within_fov = (
                context.observatory.instrument_parameters.fields_of_view / u.rad * self.star.distance).to(u.au)
        temperature_map = np.zeros(
            (len(maximum_stellar_separations_within_fov), context.settings.grid_size, context.settings.grid_size)) * u.K
        maximum_stellar_separations_radial_maps = np.zeros(
            (
                len(maximum_stellar_separations_within_fov), context.settings.grid_size,
                context.settings.grid_size)) * u.au
        mean_spectral_flux_density = np.zeros(temperature_map.shape) * u.ph / (u.m ** 2 * u.um * u.s)

        for index_separation, stellar_separation in enumerate(tqdm(maximum_stellar_separations_within_fov)):
            maximum_stellar_separations_radial_maps[index_separation] = get_radial_map(stellar_separation,
                                                                                       context.settings.grid_size)
            temperature_map[index_separation] = self._get_exozodi_temperature(
                maximum_stellar_separations_radial_maps[index_separation])

            for index_x in range(context.settings.grid_size):
                for index_y in range(context.settings.grid_size):
                    mean_spectral_flux_density[index_separation][index_x][index_y] = (create_blackbody_spectrum(
                        temperature_map[index_separation][index_x][index_y],
                        context.observatory.instrument_parameters.wavelength_range_lower_limit,
                        context.observatory.instrument_parameters.wavelength_range_upper_limit,
                        context.observatory.instrument_parameters.wavelength_bin_centers,
                        context.observatory.instrument_parameters.wavelength_bin_widths,
                        context.observatory.instrument_parameters.fields_of_view ** 2
                    )[index_separation])

        exozodi = Exozodi(**config_dict['zodi'],
                          reference_radius=np.sqrt(self.star.luminosity.to(u.Lsun)).value * u.au,
                          maximum_stellar_separations_radial_maps=maximum_stellar_separations_radial_maps)

        exozodi.mean_spectral_flux_density = mean_spectral_flux_density.reshape(
            (-1, context.settings.grid_size, context.settings.grid_size))
        return exozodi

    def _load_local_zodi(self, context: Context) -> LocalZodi:
        """Return the local zodi object from the dictionary.

        :param context: The context
        :return: The local zodi object
        """
        local_zodi = LocalZodi()
        variable_tau = 4e-8
        variable_a = 0.22
        ecliptic_latitude, relative_ecliptic_longitude = self._get_ecliptic_coordinates()

        local_zodi.mean_spectral_flux_density = (
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
        return local_zodi

    def _load_planets(self, config_dict: dict, context: Context) -> list:
        """Return the planet objects from the dictionary.

        :param config_dict: The dictionary
        :param context: The context
        :return: The planets objects
        """
        planets = []
        for key in config_dict['planets'].keys():
            planet = Planet(**config_dict['planets'][key], star_mass=self.star.mass, star_distance=self.star.distance)
            planet.mean_spectral_flux_density = create_blackbody_spectrum(planet.temperature,
                                                                          context.observatory.instrument_parameters.wavelength_range_lower_limit,
                                                                          context.observatory.instrument_parameters.wavelength_range_upper_limit,
                                                                          context.observatory.instrument_parameters.wavelength_bin_centers,
                                                                          context.observatory.instrument_parameters.wavelength_bin_widths,
                                                                          planet.solid_angle)
            planets.append(planet)
        return planets

    def _load_star(self, config_dict: dict, context: Context) -> Star:
        """Return the star object from the dictionary.

        :param config_dict: The dictionary
        :param context: The context
        :return: The star object
        """
        star = Star(**config_dict['star'])
        star.mean_spectral_flux_density = create_blackbody_spectrum(star.temperature,
                                                                    context.observatory.instrument_parameters.wavelength_range_lower_limit,
                                                                    context.observatory.instrument_parameters.wavelength_range_upper_limit,
                                                                    context.observatory.instrument_parameters.wavelength_bin_centers,
                                                                    context.observatory.instrument_parameters.wavelength_bin_widths,
                                                                    star.solid_angle)
        return star

    def apply(self, context: Context) -> Context:
        """Load the targets from the context file and initialize the star, planet and exozodi objects.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        config_dict = ConfigReader(path_to_config_file=self._path_to_context_file).get_dictionary_from_file()
        self.star = self._load_star(config_dict, context) if self.star is None else self.star
        self.planets = self._load_planets(config_dict, context) if self.planets is None else self.planets
        self.exozodi = self._load_exozodi(config_dict,
                                          context) if context.settings.noise_contributions.exozodi_leakage and self.exozodi is None else self.exozodi
        self.local_zodi = self._load_local_zodi(context) if self.local_zodi is None else self.local_zodi
        context.target_specific_photon_sources = self._add_target_specific_photon_sources(context)
        context.star_habitable_zone_central_angular_radius = self.star.habitable_zone_central_angular_radius
        return context
