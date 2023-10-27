import astropy
import numpy as np
from astropy import units as u
from tqdm import tqdm

from lifesim2.core.observation import Observation
from lifesim2.core.simulation import SimulationConfiguration
from lifesim2.util.animation import Animator


class Processor():
    """Class representing the processor. The processor makes the main calculation of the simulation.
    """

    def __init__(self, simulation_config: SimulationConfiguration, observation: Observation, animator: Animator):
        """Constructor method.

        :param simulation_config: SimulationConfiguration object
        :param observation: Observation object
        :param animator: Animator object
        """
        self.simulation_config = simulation_config
        self.observation = observation
        self.animator = animator
        self.photon_rate_time_series = dict((self.observation.sources[key].name, dict(
            (wavelength_bin_center,
             np.zeros((self.observation.observatory.beam_combination_scheme.number_of_differential_intensity_responses,
                       len(self.simulation_config.time_range)), dtype=float) * u.ph / u.s) for
            wavelength_bin_center in self.observation.observatory.instrument_parameters.wavelength_bin_centers)) for key
                                            in self.observation.sources.keys())

    def _get_differential_intensity_responses(self,
                                              time,
                                              wavelength,
                                              source_sky_coordinate_maps: np.ndarray) -> np.ndarray:
        """Return an array containing the differential intensity responses (differential virtual transmission maps), given an intensity response
         vector. For certain beam combination schemes, multiple differential intensity responses exist.

        :param time: The time for which the intensity response is calculated
        :param wavelength: The wavelength for which the intensity response is calculated
        :param source_sky_coordinate_maps: The sky coordinates of the source for which the intensity response is calculated
        :return: An array containing the differential intensity responses
        """
        intensity_response_vector = self._get_intensity_responses(time, wavelength, source_sky_coordinate_maps)
        differential_indices = self.observation.observatory.beam_combination_scheme.get_differential_intensity_response_indices()
        differential_intensity_responses = np.zeros(
            (len(differential_indices), self.simulation_config.grid_size,
             self.simulation_config.grid_size)) * intensity_response_vector.unit
        for index_index, index_pair in enumerate(differential_indices):
            differential_intensity_responses[index_index] = intensity_response_vector[index_pair[0]] - \
                                                            intensity_response_vector[
                                                                index_pair[1]]
        return differential_intensity_responses

    def _get_input_complex_amplitude_vector(self,
                                            time: astropy.units.Quantity,
                                            wavelength: astropy.units.Quantity,
                                            source_sky_coordinate_maps: np.ndarray) -> np.ndarray:
        """Return the unperturbed input complex amplitude vector, consisting of a flat wavefront per collector.

        :param time: The time for which the intensity response is calculated
        :param wavelength: The wavelength for which the intensity response is calculated
        :param source_sky_coordinate_maps: The sky coordinates of the source for which the intensity response is calculated
        :return: The input complex amplitude vector
        """
        x_observatory_coordinates, y_observatory_coordinates = self.observation.observatory.array_configuration.get_collector_positions(
            time)
        input_complex_amplitude_vector = np.zeros(
            (self.observation.observatory.beam_combination_scheme.number_of_inputs,
             self.simulation_config.grid_size, self.simulation_config.grid_size),
            dtype=complex) * self.observation.observatory.instrument_parameters.aperture_radius.unit

        for index_input in range(self.observation.observatory.beam_combination_scheme.number_of_inputs):
            input_complex_amplitude_vector[
                index_input] = self.observation.observatory.instrument_parameters.aperture_radius * np.exp(
                1j * 2 * np.pi / wavelength * (
                        x_observatory_coordinates[index_input] * source_sky_coordinate_maps[0].to(u.rad).value +
                        y_observatory_coordinates[index_input] * source_sky_coordinate_maps[1].to(u.rad).value))
        return input_complex_amplitude_vector

    def _get_intensity_responses(self,
                                 time: astropy.units.Quantity,
                                 wavelength: astropy.units.Quantity,
                                 source_sky_coordinate_maps) -> np.ndarray:
        """Return the intensity response vector.

        :param time: The time for which the intensity response is calculated
        :param wavelength: The wavelength for which the intensity response is calculated
        :param source_sky_coordinate_maps: The sky coordinates of the source for which the intensity response is calculated
        :return: The intensity response vector
        """
        input_complex_amplitude_unperturbed_vector = np.reshape(
            self._get_input_complex_amplitude_vector(time, wavelength, source_sky_coordinate_maps), (
                self.observation.observatory.beam_combination_scheme.number_of_inputs,
                self.simulation_config.grid_size ** 2))

        perturbation_matrix = self._get_perturbation_matrix()

        input_complex_amplitude_perturbed_vector = np.dot(perturbation_matrix,
                                                          input_complex_amplitude_unperturbed_vector)

        beam_combination_matrix = self.observation.observatory.beam_combination_scheme.get_beam_combination_transfer_matrix()

        intensity_response_perturbed_vector = np.reshape(abs(
            np.dot(beam_combination_matrix, input_complex_amplitude_perturbed_vector)) ** 2,
                                                         (
                                                             self.observation.observatory.beam_combination_scheme.number_of_outputs,
                                                             self.simulation_config.grid_size,
                                                             self.simulation_config.grid_size))

        return intensity_response_perturbed_vector

    def _get_perturbation_matrix(self) -> np.ndarray:
        """Return the perturbation matrix with randomly generated noise.

        :return: The perturbation matrix
        """
        if (self.simulation_config.noise_contributions.fiber_injection_variability
                or self.simulation_config.noise_contributions.optical_path_difference_variability):

            diagonal_of_matrix = []

            for index in range(self.observation.observatory.beam_combination_scheme.number_of_inputs):
                amplitude_factor = 1
                phase_difference = 0

                if self.simulation_config.noise_contributions.fiber_injection_variability:
                    amplitude_factor = np.random.uniform(0.4, 0.6)
                if self.simulation_config.noise_contributions.optical_path_difference_variability:
                    phase_difference = np.random.uniform(-0.0000000001, 0.0000000001)

                diagonal_of_matrix.append(amplitude_factor * np.exp(1j * phase_difference))

            return np.diag(diagonal_of_matrix)

        return np.diag(np.ones(self.observation.observatory.beam_combination_scheme.number_of_inputs))

    def run(self):
        """Run the main simulation time loop and calculate the photon rates time series.
        """
        for index_time, time in enumerate(tqdm(self.simulation_config.time_range)):
            for index_wavelength, wavelength in enumerate(
                    self.observation.observatory.instrument_parameters.wavelength_bin_centers):
                for _, source in self.observation.sources.items():
                    differential_intensity_responses = self._get_differential_intensity_responses(time,
                                                                                                  wavelength,
                                                                                                  source.sky_coordinate_maps)
                    for index_response, differential_intensity_response in enumerate(differential_intensity_responses):
                        self.photon_rate_time_series[source.name][wavelength][index_response][index_time] = \
                            (np.sum(differential_intensity_response * source.flux[index_wavelength] * source.shape_map *
                                    self.observation.observatory.instrument_parameters.wavelength_bin_widths[
                                        index_wavelength]))

                        if self.animator and (
                                source.name == self.animator.source_name and
                                wavelength == self.animator.closest_wavelength and
                                index_response == self.animator.differential_intensity_response_index):
                            self.animator.update_collector_position(time, self.observation)
                            self.animator.update_differential_intensity_response(differential_intensity_responses)
                            self.animator.update_photon_rate(self.output, index_time)
                            self.animator.writer.grab_frame()
