import astropy
import numpy as np
from astropy import units as u
from numpy.random import poisson, normal
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
        self.photon_count_time_series = dict((self.observation.sources[key].name, dict(
            (wavelength_bin_center,
             np.zeros((self.observation.observatory.beam_combination_scheme.number_of_differential_intensity_responses,
                       len(self.simulation_config.time_range)), dtype=float) * u.ph) for
            wavelength_bin_center in self.observation.observatory.instrument_parameters.wavelength_bin_centers)) for key
                                             in self.observation.sources.keys())
        self.intensity_response_pairs = self.observation.observatory.beam_combination_scheme.get_intensity_response_pairs()

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
        differential_indices = self.observation.observatory.beam_combination_scheme.get_intensity_response_pairs()
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
                    intensity_responses = self._get_intensity_responses(time, wavelength, source.sky_coordinate_maps)

                    for index_pair, pair_of_indices in enumerate(self.intensity_response_pairs):
                        photon_counts_at_one_output = self._get_photon_counts(
                            mean_spectral_flux_density=source.mean_spectral_flux_density[index_wavelength],
                            source_shape_map=source.shape_map,
                            wavelength_bin_width=
                            self.observation.observatory.instrument_parameters.wavelength_bin_widths[
                                index_wavelength],
                            intensity_response=intensity_responses[pair_of_indices[0]])
                        photon_counts_at_other_output = self._get_photon_counts(
                            mean_spectral_flux_density=source.mean_spectral_flux_density[index_wavelength],
                            source_shape_map=source.shape_map,
                            wavelength_bin_width=
                            self.observation.observatory.instrument_parameters.wavelength_bin_widths[
                                index_wavelength],
                            intensity_response=intensity_responses[pair_of_indices[1]])
                        self.photon_count_time_series[source.name][wavelength][index_pair][
                            index_time] = photon_counts_at_one_output - photon_counts_at_other_output

                        if self.animator and (
                                source.name == self.animator.source_name and
                                wavelength == self.animator.closest_wavelength and
                                index_pair == self.animator.differential_intensity_response_index):
                            self.animator.update_collector_position(time, self.observation)
                            self.animator.update_differential_intensity_response(
                                intensity_responses[pair_of_indices[0]] - intensity_responses[pair_of_indices[1]])
                            self.animator.update_photon_counts(
                                self.photon_count_time_series[source.name][wavelength][index_pair][
                                    index_time], index_time)
                            self.animator.writer.grab_frame()

    def _get_photon_counts(self,
                           mean_spectral_flux_density: astropy.units.Quantity,
                           source_shape_map: np.ndarray,
                           wavelength_bin_width: astropy.units.Quantity,
                           intensity_response: np.ndarray):
        """Return the photon counts per output, i.e. for a given intensity response map, including photon noise.

        :param mean_spectral_flux_density: Mean spectral density of the source at a given wavelength
        :param source_shape_map: Shape map of the source
        :param wavelength_bin_width: Wavelength bin width
        :param intensity_response: Intensity response map
        :return: Photon counts in units of photons
        """
        mean_photon_counts = np.sum((mean_spectral_flux_density * source_shape_map * wavelength_bin_width
                                     * intensity_response * self.simulation_config.time_step.to(u.s)).value)

        try:
            photon_counts = poisson(mean_photon_counts, 1)
        except ValueError:
            photon_counts = normal(mean_photon_counts, 1)
        return photon_counts * u.ph
