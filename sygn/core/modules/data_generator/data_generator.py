from enum import Enum
from random import choice
from typing import Tuple

import astropy
import numpy as np
from astropy import units as u
from numpy.random import poisson, normal
from tqdm import tqdm

from sygn.core.modules.animator.animator import Animator
from sygn.core.modules.data_generator.synthetic_data import SyntheticData
from sygn.core.modules.observation.observation import Observation
from sygn.core.modules.observatory.observatory import Observatory
from sygn.core.modules.settings.settings import Settings
from sygn.core.modules.target_system.source import Source
from sygn.core.modules.target_system.star import Star


class DataGenerationMode(Enum):
    """Enum to represent the different data generation modes.
    """
    SINGLE_OBSERVATION = 1
    YIELD_CALCULATIONS = 2


class DataGenerator():
    """Class representation of the data generator.
    """

    def __init__(self,
                 mode: DataGenerationMode,
                 settings: Settings,
                 observation: Observation,
                 observatory: Observatory,
                 target_systems: list[dict],
                 time_range: np.ndarray,
                 animator: Animator, ):
        """Constructor method.

        :param mode: The data generation mode
        :param settings: The settings object
        :param observation: The observation object
        :param observatory: The observatory object
        :param target_systems: The target systems
        :param time_range: The time range
        :param animator: The animator object
        """
        self.mode = mode
        self.settings = settings
        self.observation = observation
        self.observatory = observatory
        self.target_systems = target_systems
        self.time_range = time_range
        self.animator = animator
        # TODO: extract every target system in list
        self.output = SyntheticData(target_systems[0],
                                    observatory.instrument_parameters.wavelength_bin_centers,
                                    observatory.beam_combination_scheme.number_of_differential_intensity_responses,
                                    len(time_range))

    def _calculate_total_differential_photon_counts(self):
        """Calculate the total differential photon counts by summing over the differential photon counts of all sources.
        """
        for index_response in self.output.differential_photon_counts_by_source.keys():
            for source in self.output.differential_photon_counts_by_source[index_response].keys():
                for wavelength in self.output.differential_photon_counts_by_source[index_response][source].keys():
                    self.output.differential_photon_counts[index_response][wavelength] += \
                        self.output.differential_photon_counts_by_source[index_response][source][wavelength]

    def _create_animation(self):
        """Prepare the animation writer and run the time loop.
        """
        self.animator.prepare_animation_writer(self.target_systems,
                                               self.time_range,
                                               self.settings.grid_size)
        with self.animator.writer.saving(self.animator.figure,
                                         f'{self.animator.source_name}_{np.round(self.animator.closest_wavelength.to(u.um).value, 3)}um.gif',
                                         300):
            self._generate_differential_photon_counts()

    def _finalize_data_generation(self):
        """Finalize the data generation by calculating the total photon count time series.
        """
        self._calculate_total_differential_photon_counts()

    def _generate_differential_photon_counts(self):
        """Generate the differential photon counts. This is the main method of the data generation.
        """
        for index_time, time in enumerate(tqdm(self.time_range)):

            for index_wavelength, wavelength in enumerate(
                    self.observatory.instrument_parameters.wavelength_bin_centers):

                # TODO: do this for every target system
                for body in self.target_systems[0].values():
                    if isinstance(body, Star) and not self.settings.noise_contributions.stellar_leakage:
                        continue
                    intensity_responses = self._get_intensity_responses(time, wavelength, body.sky_coordinate_maps)

                    for index_pair, pair_of_indices in enumerate(
                            self.observatory.beam_combination_scheme.get_intensity_response_pairs()):
                        self.output.differential_photon_counts_by_source[index_pair][body.name][wavelength][
                            index_time] = self._get_differential_photon_counts(index_wavelength, body,
                                                                               intensity_responses, pair_of_indices)
                        if self.animator and (
                                body.name == self.animator.source_name and
                                wavelength == self.animator.closest_wavelength and
                                index_pair == self.animator.differential_intensity_response_index):
                            self.animator.update_collector_position(time, self.observatory)
                            self.animator.update_differential_intensity_response(
                                intensity_responses[pair_of_indices[0]] - intensity_responses[pair_of_indices[1]])
                            self.animator.update_differential_photon_counts(
                                self.output.differential_photon_counts_by_source[index_pair][body.name][wavelength][
                                    index_time], index_time)
                            self.animator.writer.grab_frame()

    def _get_differential_photon_counts(self,
                                        index_wavelength: int,
                                        source: Source,
                                        intensity_responses: np.ndarray,
                                        pair_of_indices: Tuple) -> astropy.units.Quantity:
        """Return the differential photon counts for a given wavelength, source and differential intensity response.

        :param index_wavelength: Index corresponding to the wavelength bin center
        :param source: Source object
        :param intensity_responses: Intensity response vector
        :param pair_of_indices: A pair of indices making up a differential intensity response
        :return: The differential photon counts in units  of photons
        """
        photon_counts_at_one_output = self._get_photon_counts(
            mean_spectral_flux_density=source.mean_spectral_flux_density[index_wavelength],
            source_shape_map=source.shape_map,
            wavelength_bin_width=
            self.observatory.instrument_parameters.wavelength_bin_widths[
                index_wavelength],
            intensity_response=intensity_responses[pair_of_indices[0]])
        photon_counts_at_other_output = self._get_photon_counts(
            mean_spectral_flux_density=source.mean_spectral_flux_density[index_wavelength],
            source_shape_map=source.shape_map,
            wavelength_bin_width=
            self.observatory.instrument_parameters.wavelength_bin_widths[
                index_wavelength],
            intensity_response=intensity_responses[pair_of_indices[1]])
        return photon_counts_at_one_output - photon_counts_at_other_output

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
        x_observatory_coordinates, y_observatory_coordinates = self.observatory.array_configuration.get_collector_positions(
            time)
        input_complex_amplitude_vector = np.zeros(
            (self.observatory.beam_combination_scheme.number_of_inputs,
             self.settings.grid_size, self.settings.grid_size),
            dtype=complex) * self.observatory.instrument_parameters.aperture_radius.unit

        for index_input in range(self.observatory.beam_combination_scheme.number_of_inputs):
            input_complex_amplitude_vector[
                index_input] = self.observatory.instrument_parameters.aperture_radius * np.exp(
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
                self.observatory.beam_combination_scheme.number_of_inputs,
                self.settings.grid_size ** 2))

        perturbation_matrix = self._get_perturbation_matrix(wavelength)

        input_complex_amplitude_perturbed_vector = np.dot(perturbation_matrix,
                                                          input_complex_amplitude_unperturbed_vector)

        beam_combination_matrix = self.observatory.beam_combination_scheme.get_beam_combination_transfer_matrix()

        intensity_response_perturbed_vector = np.reshape(abs(
            np.dot(beam_combination_matrix, input_complex_amplitude_perturbed_vector)) ** 2,
                                                         (
                                                             self.observatory.beam_combination_scheme.number_of_outputs,
                                                             self.settings.grid_size,
                                                             self.settings.grid_size))

        return intensity_response_perturbed_vector

    def _get_perturbation_matrix(self, wavelength: astropy.units.Quantity) -> np.ndarray:
        """Return the perturbation matrix with randomly generated noise.

        :param: Wavelength to calculate the phase error for
        :return: The perturbation matrix
        """
        if (self.settings.noise_contributions.fiber_injection_variability
                or self.settings.noise_contributions.optical_path_difference_variability):

            diagonal_of_matrix = []

            for index in range(self.observatory.beam_combination_scheme.number_of_inputs):
                amplitude_factor = 1
                phase_difference = 0 * u.um

                # TODO: Use more realistic distributions
                if self.settings.noise_contributions.fiber_injection_variability:
                    amplitude_factor = np.random.uniform(0.8, 0.9)
                if self.settings.noise_contributions.optical_path_difference_variability.apply:
                    phase_difference = choice(
                        self.settings.noise_contributions.optical_path_difference_distribution).to(u.um)

                diagonal_of_matrix.append(amplitude_factor * np.exp(2j * np.pi / wavelength * phase_difference))

            return np.diag(diagonal_of_matrix)

        return np.diag(np.ones(self.observatory.beam_combination_scheme.number_of_inputs))

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
                                     * intensity_response
                                     * self.observatory.instrument_parameters.unperturbed_instrument_throughput
                                     * self.settings.time_step.to(u.s)).value)

        try:
            photon_counts = poisson(mean_photon_counts, 1)
        except ValueError:
            photon_counts = normal(mean_photon_counts, 1)
        return photon_counts * u.ph

    def _prepare_data_generation(self):
        """Prepare the data generation.
        """
        self.observatory.array_configuration.set_optimal_baseline(self.target_systems,
                                                                  self.observation.optimized_wavelength)

    def run(self):
        """Prepare the data generation, generate the data and finalize the data generation.
        """
        self._prepare_data_generation()
        if self.animator:
            self._create_animation()
        else:
            self._generate_differential_photon_counts()
        self._finalize_data_generation()
