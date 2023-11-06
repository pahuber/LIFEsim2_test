from enum import Enum
from pathlib import Path
from typing import Any, Optional

import astropy
import numpy as np
from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from sygn.core.modules.observation.observation import Observation
from sygn.core.modules.observatory.array_configurations import ArrayConfigurationEnum, EmmaXCircularRotation, \
    EmmaXDoubleStretch, EquilateralTriangleCircularRotation, RegularPentagonCircularRotation, ArrayConfiguration
from sygn.core.modules.observatory.beam_combination_schemes import BeamCombinationScheme, BeamCombinationSchemeEnum, \
    DoubleBracewell, Kernel3, Kernel4, Kernel5
from sygn.core.modules.observatory.instrument_parameters import InstrumentParameters
from sygn.core.modules.observatory.observatory import Observatory
from sygn.core.modules.settings.noise_contributions import NoiseContributions
from sygn.core.modules.target_system.data_type import DataType
from sygn.core.modules.target_system.planet import Planet
from sygn.core.modules.target_system.star import Star
from sygn.io.config_reader import ConfigReader
from sygn.io.validators import validate_quantity_units
from sygn.util.animation import Animator
from sygn.util.blackbody import create_blackbody_spectrum
from sygn.util.grid import get_index_of_closest_value


class SimulationConfiguration(BaseModel):
    """Class representation of the simulation configurations.

    """
    grid_size: int
    time_step: Any
    noise_contributions: Optional[NoiseContributions]
    time_range: Any = None

    def __init__(self, **data):
        """Constructor method.

        :param data: Data to initialize the star class.
        """
        super().__init__(**data)
        self.noise_contributions.get_optical_path_difference_distribution(self.time_step)

    @field_validator('time_step')
    def validate_time_step(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the time step input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The time step in units of time
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=u.s)


class SimulationMode(Enum):
    """Enum to represent the different simulation modes.
    """
    SINGLE_OBSERVATION = 1
    YIELD_CALCULATIONS = 2


class Simulation():
    """Class representation of a simulation. This is the main object of this simulator.
    """

    def __init__(self):
        """Constructor method.

        :param mode: Mode of the simulation. Determines which kinds of calculations are done and what results are
                     produced
        """
        self._config_dict = None
        self.animator = None
        self.config = None
        self.observation = None

    def _initialize_array_configuration_from_config(self) -> ArrayConfiguration:
        """Return an ArrayConfiguration object.

        :return: ArrayConfiguration object.
        """
        type = self._config_dict['observatory']['array_configuration']['type']

        match type:
            case ArrayConfigurationEnum.EMMA_X_CIRCULAR_ROTATION.value:
                return EmmaXCircularRotation(**self._config_dict['observatory']['array_configuration'])

            case ArrayConfigurationEnum.EMMA_X_DOUBLE_STRETCH.value:
                return EmmaXDoubleStretch(**self._config_dict['observatory']['array_configuration'])

            case ArrayConfigurationEnum.EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION.value:
                return EquilateralTriangleCircularRotation(**self._config_dict['observatory']['array_configuration'])

            case ArrayConfigurationEnum.REGULAR_PENTAGON_CIRCULAR_ROTATION.value:
                return RegularPentagonCircularRotation(**self._config_dict['observatory']['array_configuration'])

    def _initialize_beam_combination_scheme_from_config(self) -> BeamCombinationScheme:
        """Return an BeamCombinationScheme object.

        :return: BeamCombinationScheme object.
        """
        beam_combination_scheme = self._config_dict['observatory']['beam_combination_scheme']

        match beam_combination_scheme:
            case BeamCombinationSchemeEnum.DOUBLE_BRACEWELL.value:
                return DoubleBracewell()

            case BeamCombinationSchemeEnum.KERNEL_3.value:
                return Kernel3()

            case BeamCombinationSchemeEnum.KERNEL_4.value:
                return Kernel4()

            case BeamCombinationSchemeEnum.KERNEL_5.value:
                return Kernel5()

    def _initialize_sources_from_planetary_system_configuration(self, path_to_data_file: Path):
        """Read the planetary system configuration file, extract the data and create the Star and Planet objects.

        :param path_to_data_file: Path to the data file
        """
        planetary_system_dict = ConfigReader(path_to_config_file=path_to_data_file).get_dictionary_from_file()
        star = Star(**planetary_system_dict['star'],
                    wavelength_range_lower_limit=self.observation.observatory.instrument_parameters.wavelength_range_lower_limit,
                    wavelength_range_upper_limit=self.observation.observatory.instrument_parameters.wavelength_range_upper_limit,
                    wavelength_bin_centers=self.observation.observatory.instrument_parameters.wavelength_bin_centers,
                    wavelength_bin_widths=self.observation.observatory.instrument_parameters.wavelength_bin_widths,
                    grid_size=self.config.grid_size)
        self.observation.sources[star.name] = star
        for key in planetary_system_dict['planets'].keys():
            planet = Planet(**planetary_system_dict['planets'][key],
                            star_distance=star.distance,
                            number_of_wavelength_bins=len(
                                self.observation.observatory.instrument_parameters.wavelength_bin_centers),
                            grid_size=self.config.grid_size)
            planet.mean_spectral_flux_density = create_blackbody_spectrum(planet.temperature,
                                                                          self.observation.observatory.instrument_parameters.wavelength_range_lower_limit,
                                                                          self.observation.observatory.instrument_parameters.wavelength_range_upper_limit,
                                                                          self.observation.observatory.instrument_parameters.wavelength_bin_centers,
                                                                          self.observation.observatory.instrument_parameters.wavelength_bin_widths,
                                                                          planet.solid_angle)
            self.observation.sources[planet.name] = planet

    def add_animator(self,
                     output_path: str,
                     source_name: str,
                     wavelength: astropy.units.Quantity,
                     differential_intensity_response_index: int,
                     image_vmin: float = -1,
                     image_vmax: float = 1,
                     photon_counts_limits: float = 0.1,
                     collector_position_limits: float = 50):
        """Initiate the animator object and set its attributes accordingly.

        :param output_path: Output path for the animation file
        :param source_name: Name of the source for which the animation should be made
        :param wavelength: Wavelength at which the animation should be made
        :param differential_intensity_response_index: Index specifying which of the differential outputs to animate
        :param image_vmin: Minimum value of the colormap
        :param image_vmax: Maximum value of the colormap
        :param photon_counts_limits: Limits of the photon counts plot
        :param collector_position_limits: Limits of the collector position plot
        """
        closest_wavelength = self.observation.observatory.instrument_parameters.wavelength_bin_centers[
            get_index_of_closest_value(self.observation.observatory.instrument_parameters.wavelength_bin_centers,
                                       wavelength)]
        self.animator = Animator(output_path, source_name, closest_wavelength, differential_intensity_response_index,
                                 image_vmin, image_vmax, photon_counts_limits, collector_position_limits)

    def load_sources(self, data_type: DataType, path_to_data_file: Path):
        """Add the data of a specific type.

        :param data_type: Type of the data
        :param path_to_data_file: Path to the data file
        """

        match data_type:
            case DataType.PLANETARY_SYSTEM_CONFIGURATION:
                self._initialize_sources_from_planetary_system_configuration(path_to_data_file=path_to_data_file)
            case DataType.SPECTRUM_DATA:
                # TODO: import spectral data
                pass
            case DataType.SPECTRUM_CONTEXT:
                # TODO: import spectral context data
                pass
            case DataType.POPULATION_CATALOG:
                # TODO: import population catalog
                pass

    def load_config(self, path_to_config_file: Path):
        """Extract the configuration from the file, set the parameters and instantiate the objects.

        :param path_to_config_file: Path to the configuration file
        """
        self._config_dict = ConfigReader(path_to_config_file=path_to_config_file).get_dictionary_from_file()
        self.config = SimulationConfiguration(**self._config_dict['simulation'])
        self.observation = Observation(**self._config_dict['observation'])
        self.observation.observatory = Observatory()
        self.observation.observatory.array_configuration = self._initialize_array_configuration_from_config()
        self.observation.observatory.beam_combination_scheme = self._initialize_beam_combination_scheme_from_config()
        self.observation.observatory.instrument_parameters = InstrumentParameters(
            **self._config_dict['observatory']['instrument_parameters'])
        self.config.time_range = np.arange(0, self.observation.integration_time.to(
            u.s).value, self.config.time_step.to(u.s).value) * u.s
