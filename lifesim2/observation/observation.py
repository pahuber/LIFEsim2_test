import astropy.units
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from lifesim2.observatory.observatory import Observatory
from lifesim2.util.config_reader import ConfigReader
from lifesim2.util.grid import get_meshgrid


class Observation():
    """Class representation of a simulated observation. This is the core object of the simulation process."""

    def __init__(self, path_to_config_file: str):
        self.path_to_config_file = path_to_config_file
        self.field_of_view = None
        self.integration_time = None
        self.modulation_period = None
        self.observatory = None
        self.photon_rate_time_series = []
        self.time_range = None
        self.x_sky_coordinates_map = None
        self.y_sky_coordinates_map = None

        self.simulation_grid_size = None
        self.simulation_time_step = None

        self._set_from_config()

    def _set_from_config(self):
        """Read the configuration file and set the Observation parameters.
        """
        self._config_dict = ConfigReader(path_to_config_file=self.path_to_config_file).get_config_from_file()
        self.field_of_view = 0.4 * u.arcsec  # TODO: calculate fov
        self.integration_time = self._config_dict['observation']['integration_time']
        self.modulation_period = self._config_dict['observatory']['array_configuration']['modulation_period']
        self.simulation_grid_size = int(self._config_dict['simulation']['grid_size'].value)
        self.simulation_time_step = self._config_dict['simulation']['time_step']

        self.time_range = np.arange(0, self.modulation_period.to(u.s).value,
                                    self.simulation_time_step.to(u.s).value) * u.s
        self.x_sky_coordinates_map, self.y_sky_coordinates_map = get_meshgrid(self.field_of_view,
                                                                              self.simulation_grid_size)

        self.create_observatory()

    def create_observatory(self):
        """Create an observatory object from the configuration file.
        """
        self.observatory = Observatory()
        self.observatory.set_from_config(observatory_dictionary=self._config_dict['observatory'])

    def run(self):
        """Main method of the simulator. Run the simulated observation and calculate the photon rate time series.
        """
        beam_combination_matrix = self.observatory.beam_combination_scheme.get_beam_combination_transfer_matrix()

        for time_index, time in enumerate(self.time_range):
            # Get intensity response vector and virtual transmission maps
            input_complex_amplitude_unperturbed_vector = np.reshape(self.get_input_complex_amplitude_vector(time), (
                self.observatory.beam_combination_scheme.number_of_inputs, self.simulation_grid_size ** 2))

            perturbation_matrix = self.get_perturbation_matrix()

            input_complex_amplitude_perturbed_vector = np.dot(perturbation_matrix,
                                                              input_complex_amplitude_unperturbed_vector)

            intensity_response_vector = np.reshape(abs(
                np.dot(beam_combination_matrix, input_complex_amplitude_perturbed_vector)) ** 2,
                                                   (
                                                       self.observatory.beam_combination_scheme.number_of_outputs,
                                                       self.simulation_grid_size,
                                                       self.simulation_grid_size))

            t_map = np.real(intensity_response_vector[2] - intensity_response_vector[3])
            # 
            # plt.imshow(t_map, vmin=-1.6, vmax=1.6)
            # plt.colorbar()
            # plt.savefig(f't_{time_index}.png')
            # # plt.show()
            # plt.close()

            # Given the transmission maps, get the photon rate per source
            self.photon_rate_time_series.append(t_map[self.simulation_grid_size // 4][self.simulation_grid_size // 4])

        # Save the photon rate time series
        plt.plot(self.photon_rate_time_series)
        plt.savefig('photon_rate.png')
        plt.close()

    def get_input_complex_amplitude_vector(self, time: astropy.units.Quantity) -> np.ndarray:
        """Return the unperturbed input complex amplitude vector, consisting of a flat wavefront per collector.

        :param time: The time to calculate the vector at
        :return: The input complex amplitude vector
        """
        wavelength = 10 * u.um
        x_sky_coordinates = self.x_sky_coordinates_map.to(u.rad).value
        y_sky_coordinates = self.y_sky_coordinates_map.to(u.rad).value

        x_observatory_coordinates, y_observatory_coordinates = self.observatory.array_configuration.get_collector_positions(
            time)

        input_complex_amplitude_vector = np.zeros((self.observatory.beam_combination_scheme.number_of_inputs,
                                                   self.simulation_grid_size, self.simulation_grid_size), dtype=complex)

        for index_input in range(self.observatory.beam_combination_scheme.number_of_inputs):
            input_complex_amplitude_vector[index_input] = (
                    self.observatory.instrument_parameters.aperture_radius * np.exp(
                1j * 2 * np.pi / wavelength * (
                        x_observatory_coordinates[index_input] * x_sky_coordinates + y_observatory_coordinates[
                    index_input] * y_sky_coordinates)))

        return input_complex_amplitude_vector

    def get_perturbation_matrix(self) -> np.ndarray:
        """Return the perturbation matrix with randomly generated noise.

        :return: The perturbation matrix
        """
        diagonal_of_matrix = []
        for index in range(self.observatory.beam_combination_scheme.number_of_inputs):
            diagonal_of_matrix.append(np.random.uniform(0.2, 0.4) * np.exp(1j * np.random.uniform(-0.2, 0.2)))

        perturbation_matrix = np.diag(diagonal_of_matrix)
        # perturbation_matrix = np.diag([1, 1, 1])

        return perturbation_matrix
