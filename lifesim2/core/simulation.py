from enum import Enum

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from lifesim2.core.data import DataType
from lifesim2.core.observation.observation import Observation
from lifesim2.core.observation.observatory.array_configurations import ArrayConfigurationEnum, EmmaXCircularRotation, \
    EmmaXDoubleStretch, EquilateralTriangleCircularRotation, RegularPentagonCircularRotation
from lifesim2.core.observation.observatory.beam_combination_schemes import BeamCombinationSchemeEnum, DoubleBracewell, \
    Kernel3, \
    Kernel4, Kernel5
from lifesim2.core.observation.observatory.instrument_parameters import InstrumentParameters
from lifesim2.core.observation.observatory.observatory import Observatory
from lifesim2.core.observation.sources.general import get_input_complex_amplitude_vector, get_perturbation_matrix
from lifesim2.util.config_reader import ConfigReader


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
        self._configurations = None
        self.grid_size = None
        self.time_step = None
        self.observation = None
        self.photon_rate_time_series = []

    def run(self):
        """Main method of the simulator. Run the simulated observation and calculate the photon rate time series.
        """
        beam_combination_matrix = self.observation.observatory.beam_combination_scheme.get_beam_combination_transfer_matrix()

        for time_index, time in enumerate(self.time_range):
            # Get intensity response vector and virtual transmission maps
            input_complex_amplitude_unperturbed_vector = np.reshape(
                get_input_complex_amplitude_vector(self.observation, time), (
                    self.observation.observatory.beam_combination_scheme.number_of_inputs, self.grid_size ** 2))

            perturbation_matrix = get_perturbation_matrix(self.observation)

            input_complex_amplitude_perturbed_vector = np.dot(perturbation_matrix,
                                                              input_complex_amplitude_unperturbed_vector)

            intensity_response_vector = np.reshape(abs(
                np.dot(beam_combination_matrix, input_complex_amplitude_perturbed_vector)) ** 2,
                                                   (
                                                       self.observation.observatory.beam_combination_scheme.number_of_outputs,
                                                       self.grid_size,
                                                       self.grid_size))

            t_map = np.real(intensity_response_vector[2] - intensity_response_vector[3])
            #
            plt.imshow(t_map, vmin=-1.6, vmax=1.6)
            plt.colorbar()
            plt.savefig(f't_{time_index}.png')
            # plt.show()
            plt.close()

            # Given the transmission maps, get the photon rate per source
            self.photon_rate_time_series.append(t_map[self.grid_size // 4][self.grid_size // 4])

        # Save the photon rate time series
        plt.plot(self.photon_rate_time_series)
        plt.savefig('photon_rate.png')
        plt.show()

    def load_config(self, path_to_config_file):
        """Extract the configuration from the file, set the parameters and instantiate the objects.

        :param path_to_config_file: Path to the configuration file
        """
        self._configurations = ConfigReader(path_to_config_file=path_to_config_file).get_config_from_file()
        self._load_simulation_config()
        self.observation = self._create_observation_from_config()
        self.observation.observatory = self._create_observatory_from_config()
        self.observation.observatory.array_configuration = self._create_array_configuration_from_config()
        self.observation.observatory.beam_combination_scheme = self._create_beam_combination_scheme_from_config()
        self.observation.observatory.instrument_parameters = self._create_instrument_parameters_from_config()
        self._create_composite_variables()

    def import_data(self, type: DataType, path_to_data_file: str):
        """Import the data of a specific type.

        :param type: Type of the data
        :param path_to_data_file: Path to the data file
        """

        match type.value:
            case 1:
                # TODO: generate planetary blackbody spectra
                pass
            case 2:
                # TODO: import spectral data
                pass
            case 3:
                # TODO: import spectral context data
                pass
            case 4:
                # TODO: import population catalog
                pass

    def _load_simulation_config(self):
        """Set the class variables from the configuration.
        """
        self.grid_size = int(self._configurations['simulation']['grid_size'].value)
        self.time_step = self._configurations['simulation']['time_step']

    def _create_observation_from_config(self):
        """Return an observation object.

        :return: Observation object.
        """
        return Observation(
            adjust_baseline_to_habitable_zone=self._configurations['observation']['adjust_baseline_to_habitable_zone'],
            integration_time=self._configurations['observation']['integration_time'],
            optimized_wavelength=self._configurations['observation']['optimized_wavelength'],
            grid_size=self.grid_size)

    def _create_observatory_from_config(self):
        """Return an observatory object.

        :return: Observatory object.
        """
        return Observatory()

    def _create_array_configuration_from_config(self):
        """Return an ArrayConfiguration object.

        :return: ArrayConfiguration object.
        """
        array_configuration = self._configurations['observatory']['array_configuration']['type']
        baseline_minimum = self._configurations['observatory']['array_configuration']['baseline_minimum']
        baseline_maximum = self._configurations['observatory']['array_configuration']['baseline_maximum']
        baseline_ratio = self._configurations['observatory']['array_configuration']['baseline_ratio']
        modulation_period = self._configurations['observatory']['array_configuration']['modulation_period']

        match array_configuration:
            case ArrayConfigurationEnum.EMMA_X_CIRCULAR_ROTATION.value:
                return EmmaXCircularRotation(baseline_minimum=baseline_minimum,
                                             baseline_maximum=baseline_maximum,
                                             baseline_ratio=baseline_ratio,
                                             modulation_period=modulation_period)

            case ArrayConfigurationEnum.EMMA_X_DOUBLE_STRETCH.value:
                return EmmaXDoubleStretch(baseline_minimum=baseline_minimum,
                                          baseline_maximum=baseline_maximum,
                                          baseline_ratio=baseline_ratio,
                                          modulation_period=modulation_period)

            case ArrayConfigurationEnum.EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION.value:
                return EquilateralTriangleCircularRotation(baseline_minimum=baseline_minimum,
                                                           baseline_maximum=baseline_maximum,
                                                           baseline_ratio=baseline_ratio,
                                                           modulation_period=modulation_period)

            case ArrayConfigurationEnum.REGULAR_PENTAGON_CIRCULAR_ROTATION.value:
                return RegularPentagonCircularRotation(baseline_minimum=baseline_minimum,
                                                       baseline_maximum=baseline_maximum,
                                                       baseline_ratio=baseline_ratio,
                                                       modulation_period=modulation_period)

    def _create_beam_combination_scheme_from_config(self):
        """Return an BeamCombinationScheme object.

        :return: BeamCombinationScheme object.
        """
        beam_combination_scheme = self._configurations['observatory']['beam_combination_scheme']

        match beam_combination_scheme:
            case BeamCombinationSchemeEnum.DOUBLE_BRACEWELL.value:
                return DoubleBracewell()

            case BeamCombinationSchemeEnum.KERNEL_3.value:
                return Kernel3()

            case BeamCombinationSchemeEnum.KERNEL_4.value:
                return Kernel4()

            case BeamCombinationSchemeEnum.KERNEL_5.value:
                return Kernel5()

    def _create_instrument_parameters_from_config(self):
        """Return an InstrumentParameters object.

        :return: InstrumentParameters object.
        """
        return InstrumentParameters(
            aperture_diameter=self._configurations['observatory']['instrument_parameters']['aperture_diameter'],
            spectral_range_lower_limit=self._configurations['observatory']['instrument_parameters'][
                'spectral_range_lower_limit'],
            spectral_range_upper_limit=self._configurations['observatory']['instrument_parameters'][
                'spectral_range_upper_limit'],
            spectral_resolution=self._configurations['observatory']['instrument_parameters']['spectral_resolution'])

    def _create_composite_variables(self):
        """Create and set composite variables.
        """
        self.time_range = np.arange(0, self.observation.observatory.array_configuration.modulation_period.to(u.s).value,
                                    self.time_step.to(u.s).value) * u.s
