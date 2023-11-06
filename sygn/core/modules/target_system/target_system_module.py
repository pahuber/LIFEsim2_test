from pathlib import Path

from sygn.core.context import Context
from sygn.core.modules.base_module import BaseModule
from sygn.core.modules.target_system.data_type import DataType
from sygn.core.modules.target_system.planet import Planet
from sygn.core.modules.target_system.star import Star
from sygn.io.config_reader import ConfigReader
from sygn.util.blackbody import create_blackbody_spectrum


class TargetSystemModule(BaseModule):
    def __init__(self, path_to_data_file: Path, data_type: DataType):
        self.path_to_data_file = path_to_data_file
        self.data_type = data_type
        self.target_system = {}

    def _initialize_sources_from_planetary_system_configuration(self, context):
        """Read the planetary system configuration file, extract the data and create the Star and Planet objects.

        :param path_to_data_file: Path to the data file
        """
        configuration_dict = ConfigReader(path_to_config_file=self.path_to_data_file).get_dictionary_from_file()
        star = Star(**configuration_dict['star'],
                    wavelength_range_lower_limit=context.observatory.instrument_parameters.wavelength_range_lower_limit,
                    wavelength_range_upper_limit=context.observatory.instrument_parameters.wavelength_range_upper_limit,
                    wavelength_bin_centers=context.observatory.instrument_parameters.wavelength_bin_centers,
                    wavelength_bin_widths=context.observatory.instrument_parameters.wavelength_bin_widths,
                    grid_size=context.settings.grid_size)
        self.target_system[star.name] = star
        for key in configuration_dict['planets'].keys():
            planet = Planet(**configuration_dict['planets'][key],
                            star_distance=star.distance,
                            number_of_wavelength_bins=len(
                                context.observatory.instrument_parameters.wavelength_bin_centers),
                            grid_size=context.settings.grid_size)
            planet.mean_spectral_flux_density = create_blackbody_spectrum(planet.temperature,
                                                                          context.observatory.instrument_parameters.wavelength_range_lower_limit,
                                                                          context.observatory.instrument_parameters.wavelength_range_upper_limit,
                                                                          context.observatory.instrument_parameters.wavelength_bin_centers,
                                                                          context.observatory.instrument_parameters.wavelength_bin_widths,
                                                                          planet.solid_angle)
            self.target_system[planet.name] = planet

    def _load_sources(self, context):
        """Add the data of a specific type.

        :param data_type: Type of the data
        :param path_to_data_file: Path to the data file
        """

        match self.data_type:
            case DataType.PLANETARY_SYSTEM_CONFIGURATION:
                self._initialize_sources_from_planetary_system_configuration(context)
            case DataType.SPECTRUM_DATA:
                # TODO: import spectral data
                pass
            case DataType.SPECTRUM_CONTEXT:
                # TODO: import spectral context data
                pass
            case DataType.POPULATION_CATALOG:
                # TODO: import population catalog
                pass

    def apply(self, context: Context) -> Context:
        self._load_sources(context)
        context.target_systems.append(self.target_system)
        return context
