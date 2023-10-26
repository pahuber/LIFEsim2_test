from enum import Enum

import astropy
import numpy as np
from astropy import units as u

from lifesim2.core.observation import Observation
from lifesim2.core.observatory.array_configurations import ArrayConfigurationEnum, EmmaXCircularRotation, \
    EmmaXDoubleStretch, EquilateralTriangleCircularRotation, RegularPentagonCircularRotation, ArrayConfiguration
from lifesim2.core.observatory.beam_combination_schemes import BeamCombinationSchemeEnum, DoubleBracewell, \
    Kernel3, \
    Kernel4, Kernel5, BeamCombinationScheme
from lifesim2.core.observatory.instrument_parameters import InstrumentParameters
from lifesim2.core.observatory.observatory import Observatory
from lifesim2.core.simulation_configuration import SimulationConfiguration
from lifesim2.core.sources.planet import Planet
from lifesim2.core.sources.star import Star
from lifesim2.io.config_reader import ConfigReader
from lifesim2.io.data_type import DataType
from lifesim2.io.simulation_output import SimulationOutput
from lifesim2.processing.processor import Processor
from lifesim2.util.animation import Animator
from lifesim2.util.blackbody import create_blackbody_spectrum
from lifesim2.util.grid import get_index_of_closest_value


class SimulationMode(Enum):
    """Enum to represent the different simulation modes.
    """
    SINGLE_OBSERVATION = 1
    YIELD_ESTIMATE = 2


class Simulation():
    """Class representation of a simulation. This is the main object of this simulator.
    """

    def __init__(self, mode: SimulationMode):
        """Constructor method.

        :param mode: Mode of the simulation. Determines which kinds of calculations are done and what results are
                     produced
        """
        self.mode = mode
        self._config_dict = None
        self.animator = None
        self.config = None
        self.observation = None
        self.processor = None

    def _create_animation(self):
        """Prepare the animation writer and run the time loop.
        """
        self.animator.prepare_animation_writer(self.observation, self.config.time_range, self.config.grid_size)
        with self.animator.writer.saving(self.animator.figure,
                                         f'{self.animator.source_name}_{np.round(self.animator.closest_wavelength.to(u.um).value, 3)}um.gif',
                                         300):
            self.processor.run()

    def _finish_run(self):
        """Finish the run by calculating the total photon rate time series.
        """
        self.output.photon_rate_time_series = self.processor.photon_rate_time_series
        self.output._calculate_total_photon_rate_time_series()
        a = 0

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

    def _initialize_sources_from_planetary_system_configuration(self, path_to_data_file: str):
        """Read the planetary system configuration file, extract the data and create the Star and Planet objects.

        :param path_to_data_file: Path to the data file
        """
        planetary_system_dict = ConfigReader(path_to_config_file=path_to_data_file).get_config_from_file()
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
            planet.flux = create_blackbody_spectrum(planet.temperature,
                                                    self.observation.observatory.instrument_parameters.wavelength_range_lower_limit,
                                                    self.observation.observatory.instrument_parameters.wavelength_range_upper_limit,
                                                    self.observation.observatory.instrument_parameters.wavelength_bin_centers,
                                                    self.observation.observatory.instrument_parameters.wavelength_bin_widths,
                                                    planet.solid_angle)
            self.observation.sources[planet.name] = planet

    def _prepare_run(self):
        """Prepare the main simulation run.
        """
        self.processor = Processor(self.config, self.observation, self.animator)
        self.observation.set_optimal_baseline()
        self.output = SimulationOutput(
            self.observation.observatory.beam_combination_scheme.number_of_differential_intensity_responses,
            len(self.config.time_range),
            self.observation.observatory.instrument_parameters.wavelength_bin_centers,
            self.observation.sources)

    def animate(self,
                output_path: str,
                source_name: str,
                wavelength: astropy.units.Quantity,
                differential_intensity_response_index: int,
                image_vmin: float = -1,
                image_vmax: float = 1,
                photon_rate_limits: float = 0.1,
                collector_position_limits: float = 50):
        """Initiate the animator object and set its attributes accordingly.

        :param output_path: Output path for the animation file
        :param source_name: Name of the source for which the animation should be made
        :param wavelength: Wavelength at which the animation should be made
        :param differential_intensity_response_index: Index specifying which of the differential outputs to animate
        :param image_vmin: Minimum value of the colormap
        :param image_vmax: Maximum value of the colormap
        :param photon_rate_limits: Limits of the photon rate plot
        :param collector_position_limits: Limits of the collector position plot
        """
        closest_wavelength = self.observation.observatory.instrument_parameters.wavelength_bin_centers[
            get_index_of_closest_value(self.observation.observatory.instrument_parameters.wavelength_bin_centers,
                                       wavelength)]
        self.animator = Animator(output_path, source_name, closest_wavelength, differential_intensity_response_index,
                                 image_vmin, image_vmax, photon_rate_limits, collector_position_limits)

    def import_data(self, type: DataType, path_to_data_file: str):
        """Import the data of a specific type.

        :param type: Type of the data
        :param path_to_data_file: Path to the data file
        """

        match type:
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

    def load_config(self, path_to_config_file):
        """Extract the configuration from the file, set the parameters and instantiate the objects.

        :param path_to_config_file: Path to the configuration file
        """
        self._config_dict = ConfigReader(path_to_config_file=path_to_config_file).get_config_from_file()
        self.config = SimulationConfiguration(**self._config_dict['simulation'])
        self.observation = Observation(**self._config_dict['observation'])
        self.observation.observatory = Observatory()
        self.observation.observatory.array_configuration = self._initialize_array_configuration_from_config()
        self.observation.observatory.beam_combination_scheme = self._initialize_beam_combination_scheme_from_config()
        self.observation.observatory.instrument_parameters = InstrumentParameters(
            **self._config_dict['observatory']['instrument_parameters'])
        self.config.time_range = np.arange(0, self.observation.integration_time.to(
            u.s).value, self.config.time_step.to(u.s).value) * u.s

    def run(self):
        """Prepare and run the main simulation.
        """
        self._prepare_run()
        if self.animator:
            self._create_animation()
        else:
            self.processor.run()
        self._finish_run()
