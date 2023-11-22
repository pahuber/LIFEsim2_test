from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic

from sygn.core.context import Context
from sygn.core.entities.photon_sources.exozodi import Exozodi
from sygn.core.entities.photon_sources.local_zodi import LocalZodi
from sygn.core.entities.photon_sources.planet import Planet
from sygn.core.entities.photon_sources.star import Star
from sygn.io.config_reader import ConfigReader
from sygn.util.blackbody import create_blackbody_spectrum


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

    def _add_target_specific_photon_sources(self, context):
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

    def _load_exozodi(self, config_dict: dict) -> Exozodi:
        """Return the exozodi object from the dictionary.

        :param config_dict: The dictionary
        :return: The exozodi object
        """
        return Exozodi(**config_dict['zodi'])

    def _load_local_zodi(self, context: Context) -> LocalZodi:
        """Return the local zodi object from the dictionary.

        :return: The local zodi object
        """
        local_zodi = LocalZodi()
        variable_tau = 4e-8
        variable_a = 0.22
        coordinates = SkyCoord(ra=self.star.right_ascension, dec=self.star.declination, frame='icrs')
        coordinates_ecliptic = coordinates.transform_to(GeocentricTrueEcliptic)
        ecliptic_latitude = coordinates_ecliptic.lat.to(u.deg)
        ecliptic_longitude = coordinates_ecliptic.lon.to(u.deg)
        relative_ecliptic_longitude = ecliptic_longitude - 0 * u.deg
        # a = {'name': 'Sun',
        #      'temperature': 5778 * u.K,
        #      'radius': 1 * u.Rsun,
        #      'mass': 1 * u.Msun,
        #      'distance': 1 * u.au,
        #      'luminosity': 1 * u.Lsun,
        #      'right_ascension': 0 * u.deg,
        #      'declination': 0 * u.deg}
        # sun = Star(**a)

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
        self.exozodi = self._load_exozodi(config_dict) if self.exozodi is None else self.exozodi
        self.local_zodi = self._load_local_zodi(context) if self.local_zodi is None else self.local_zodi
        context.target_specific_photon_sources = self._add_target_specific_photon_sources(context)
        context.star_habitable_zone_central_angular_radius = self.star.habitable_zone_central_angular_radius
        return context
