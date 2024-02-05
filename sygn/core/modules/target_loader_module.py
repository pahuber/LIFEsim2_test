from pathlib import Path

from sygn.core.context import Context
from sygn.core.entities.photon_sources.exozodi import Exozodi
from sygn.core.entities.photon_sources.local_zodi import LocalZodi
from sygn.core.entities.photon_sources.photon_source import PhotonSource
from sygn.core.entities.photon_sources.planet import Planet
from sygn.core.entities.photon_sources.star import Star
from sygn.core.modules.base_module import BaseModule
from sygn.io.config_reader import ConfigReader


class TargetLoaderModule(BaseModule):
    """Class representation of the target loader module.
    """

    def __init__(self,
                 path_to_context_file: Path,
                 path_to_spectrum_file: Path = None,
                 config_dict: dict = None):
        """Constructor method.

        :param path_to_context_file: Path to the context file
        :param path_to_spectrum_file: Path to the spectrum file
        """
        self._path_to_context_file = path_to_context_file
        self._path_to_spectrum_file = path_to_spectrum_file
        self._config_dict = config_dict
        self.star = None
        self.planets = None
        self.exozodi = None
        self.local_zodi = None
        self.dependencies = []

    def _add_target_specific_photon_sources(self, context) -> list:
        """Add the photon sources to the list, if specified in the configurations.

        :param context: The context
        :return: The list containing the specified photon sources
        """
        target_specific_photon_sources = []
        if context.settings.noise.stellar_leakage:
            target_specific_photon_sources.append(self.star)
        if context.settings.noise.local_zodi_leakage:
            target_specific_photon_sources.append(
                self.local_zodi)
        if context.settings.noise.exozodi_leakage:
            target_specific_photon_sources.append(self.exozodi)
        for planet in self.planets:
            target_specific_photon_sources.append(planet)
        return target_specific_photon_sources

    def _load_exozodi(self, config_dict: dict, context: Context) -> Exozodi:
        """Return the exozodi object from the dictionary.

        :param config_dict: The dictionary
        :param context: The context
        :return: The exozodi object
        """
        if self.exozodi:
            exozodi = self._setup_source(self.exozodi, context)
        else:
            exozodi = Exozodi(**config_dict['zodi'],
                              star_distance=self.star.distance,
                              star_luminosity=self.star.luminosity)
            exozodi = self._setup_source(exozodi, context)
        return exozodi

    def _load_local_zodi(self, context: Context) -> LocalZodi:
        """Return the local zodi object from the dictionary.

        :param context: The context
        :return: The local zodi object
        """
        if self.local_zodi:
            local_zodi = self._setup_source(self.local_zodi, context)
        else:
            local_zodi = LocalZodi(star_right_ascension=self.star.right_ascension,
                                   star_declination=self.star.declination)
            local_zodi = self._setup_source(local_zodi, context)
        return local_zodi

    def _load_planets(self, config_dict: dict, context: Context) -> list:
        """Return the planet objects from the dictionary.

        :param config_dict: The dictionary
        :param context: The context
        :return: The planets objects
        """
        planets = []

        # If the user directly provides a list of planet objects
        if self.planets:
            for planet in self.planets:
                planet = self._setup_source(planet, context)
                planets.append(planet)

        # If the planet objects have to be loaded from the context file
        else:
            for key in config_dict['planets'].keys():
                planet = Planet(**config_dict['planets'][key], star_mass=self.star.mass,
                                star_distance=self.star.distance)
                planet = self._setup_source(planet, context)
                planets.append(planet)
        return planets

    def _load_star(self, config_dict: dict, context: Context) -> Star:
        """Return the star object from the dictionary.

        :param config_dict: The dictionary
        :param context: The context
        :return: The star object
        """
        if self.star:
            star = self._setup_source(self.star, context)
        else:
            star = Star(**config_dict['star'])
            star = self._setup_source(star, context)
        return star

    def _setup_source(self, source: PhotonSource, context: Context) -> PhotonSource:
        source.setup(context=context)
        return source

    def apply(self, context: Context) -> Context:
        """Load the targets from the context file and initialize the star, planet and exozodi objects.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        if not self._config_dict:
            self._config_dict = ConfigReader(path_to_config_file=self._path_to_context_file).get_dictionary_from_file()
        self.star = self._load_star(self._config_dict, context)
        self.planets = self._load_planets(self._config_dict, context)
        self.exozodi = self._load_exozodi(self._config_dict,
                                          context) if context.settings.noise.exozodi_leakage else None
        self.local_zodi = self._load_local_zodi(
            context) if context.settings.noise.local_zodi_leakage else None
        context.photon_sources = self._add_target_specific_photon_sources(context)
        context.star = self.star
        return context
