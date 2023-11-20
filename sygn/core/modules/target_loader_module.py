from pathlib import Path

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

    def _load_local_zodi(self) -> LocalZodi:
        """Return the local zodi object from the dictionary.

        :return: The local zodi object
        """
        return LocalZodi(star_right_ascension=self.star.right_ascension, star_declination=self.star.declination)

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
        self.local_zodi = self._load_local_zodi() if self.local_zodi is None else self.local_zodi
        context.target_specific_photon_sources = self._add_target_specific_photon_sources(context)
        context.star_habitable_zone_central_angular_radius = self.star.habitable_zone_central_angular_radius
        return context
