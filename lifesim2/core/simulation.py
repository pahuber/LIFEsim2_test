from enum import Enum

import numpy as np
from astropy import units as u

from lifesim2.core.intensity_response import get_differential_intensity_responses
from lifesim2.core.observation import Observation
from lifesim2.core.observatory.array_configurations import ArrayConfigurationEnum, EmmaXCircularRotation, \
    EmmaXDoubleStretch, EquilateralTriangleCircularRotation, RegularPentagonCircularRotation
from lifesim2.core.observatory.beam_combination_schemes import BeamCombinationSchemeEnum, DoubleBracewell, \
    Kernel3, \
    Kernel4, Kernel5
from lifesim2.core.observatory.instrument_parameters import InstrumentParameters
from lifesim2.core.observatory.observatory import Observatory
from lifesim2.core.sources.planet import Planet
from lifesim2.core.sources.star import Star
from lifesim2.io.config_reader import ConfigReader
from lifesim2.io.data_type import DataType
from lifesim2.io.simulation_output import SimulationOutput
from lifesim2.util.blackbody import create_blackbody_spectrum
from lifesim2.util.grid import get_sky_coordinates


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

    def run(self):
        """Main method of the simulator. Run the simulated observation and calculate the photon rate time series.
        """
        beam_combination_matrix = self.observation.observatory.beam_combination_scheme.get_beam_combination_transfer_matrix()
        self.output = SimulationOutput(self.observation.observatory.beam_combination_scheme.number_of_transmission_maps,
                                       len(self.time_range),
                                       self.observation.observatory.instrument_parameters.wavelength_bin_centers)

        for time_index, time in enumerate(self.time_range):
            for wavelength_index, wavelength in enumerate(
                    self.observation.observatory.instrument_parameters.wavelength_bin_centers):
                differential_intensity_responses = get_differential_intensity_responses(time,
                                                                                        wavelength,
                                                                                        self.observation,
                                                                                        self.grid_size)

                # TODO: For each source, calculate photon rate
                for source in self.observation.sources:
                    if isinstance(source, Planet):
                        print(source.name, wavelength, source.flux[
                            wavelength_index])
                    # get flux per bin
                    pass
                    # if flux is for one pixel, multiply by flux location on transmission map
                    # integrate all fluxes
                    # add to total photon rate and individual source photon rate
                self.output.append_photon_rate(time_index, differential_intensity_responses, wavelength)

                # plt.imshow(differential_intensity_responses[0])
                # plt.colorbar()
                # # plt.savefig(f't_{time_index}.png')
                # plt.show()
                # plt.close()

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

        match type:
            case DataType.PLANETARY_SYSTEM_CONFIGURATION:
                self._create_sources_from_planetary_system_configuration(path_to_data_file=path_to_data_file)
            case DataType.SPECTRUM_DATA:
                # TODO: import spectral data
                pass
            case DataType.SPECTRUM_CONTEXT:
                # TODO: import spectral context data
                pass
            case DataType.POPULATION_CATALOG:
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
            wavelength_range_lower_limit=self._configurations['observatory']['instrument_parameters'][
                'wavelength_range_lower_limit'],
            wavelength_range_upper_limit=self._configurations['observatory']['instrument_parameters'][
                'wavelength_range_upper_limit'],
            spectral_resolving_power=self._configurations['observatory']['instrument_parameters'][
                'spectral_resolving_power'])

    def _create_composite_variables(self):
        """Create and set some composite variables.
        """
        self.time_range = np.arange(0, self.observation.observatory.array_configuration.modulation_period.to(u.s).value,
                                    self.time_step.to(u.s).value) * u.s
        # TODO: implement baseline correctly
        self.observation.observatory.instrument_parameters.field_of_view = list(((wavelength.to(
            u.m) / self.observation.observatory.array_configuration.baseline_minimum.to(u.m)) * u.rad).to(u.arcsec) for
                                                                                wavelength in
                                                                                self.observation.observatory.instrument_parameters.wavelength_bin_centers)
        self.observation.observatory.x_sky_coordinates_map, self.observation.observatory.y_sky_coordinates_map = get_sky_coordinates(
            self.observation.observatory.instrument_parameters.wavelength_bin_centers,
            self.observation.observatory.instrument_parameters.field_of_view, self.grid_size)
        self.angle_per_pixel = list(field_of_view / self.grid_size for field_of_view in
                                    self.observation.observatory.instrument_parameters.field_of_view)

    def _create_sources_from_planetary_system_configuration(self, path_to_data_file: str):
        """Read the planetary system configuration file, extract the data and create the Star and Planet objects.

        :param path_to_data_file: Path to the data file
        """
        planetary_system_dict = ConfigReader(path_to_config_file=path_to_data_file).get_config_from_file()
        star = Star(name=planetary_system_dict['star']['name'],
                    radius=planetary_system_dict['star']['radius'],
                    temperature=planetary_system_dict['star']['temperature'],
                    mass=planetary_system_dict['star']['mass'],
                    distance=planetary_system_dict['star']['distance'],
                    number_of_wavelength_bins=len(
                        self.observation.observatory.instrument_parameters.wavelength_bin_centers))
        star.flux = create_blackbody_spectrum(star.temperature,
                                              self.observation.observatory.instrument_parameters.wavelength_range_lower_limit,
                                              self.observation.observatory.instrument_parameters.wavelength_range_upper_limit,
                                              self.observation.observatory.instrument_parameters.wavelength_bin_centers,
                                              self.observation.observatory.instrument_parameters.wavelength_bin_widths,
                                              self.angle_per_pixel)
        self.observation.sources.append(star)
        for key in planetary_system_dict['planets'].keys():
            planet = Planet(name=planetary_system_dict['planets'][key]['name'],
                            radius=planetary_system_dict['planets'][key]['radius'],
                            temperature=planetary_system_dict['planets'][key]['temperature'],
                            mass=planetary_system_dict['planets'][key]['mass'],
                            star_separation=planetary_system_dict['planets'][key]['star_separation'],
                            star_distance=star.distance,
                            number_of_wavelength_bins=len(
                                self.observation.observatory.instrument_parameters.wavelength_bin_centers))
            planet.flux = create_blackbody_spectrum(planet.temperature,
                                                    self.observation.observatory.instrument_parameters.wavelength_range_lower_limit,
                                                    self.observation.observatory.instrument_parameters.wavelength_range_upper_limit,
                                                    self.observation.observatory.instrument_parameters.wavelength_bin_centers,
                                                    self.observation.observatory.instrument_parameters.wavelength_bin_widths,
                                                    self.angle_per_pixel)
            self.observation.sources.append(planet)
