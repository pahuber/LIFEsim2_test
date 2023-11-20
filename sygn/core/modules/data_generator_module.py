from random import choice

import astropy
import numpy as np
from astropy import units as u
from numpy.random import poisson, normal
from tqdm.contrib.itertools import product

from sygn.core.context import Context
from sygn.core.modules.base_module import BaseModule
from sygn.util.helpers import Coordinates


class DataGeneratorModule(BaseModule):
    """Class representation of the data generator modules.
    """

    def __init__(self):
        """Constructor method.

        :param path_to_config_file: Path to the config file
        """

    def _set_optimal_baseline(self, context: Context):
        context.observatory.array_configuration.set_optimal_baseline(
            context.mission.optimized_wavelength,
            context.star_habitable_zone_central_angular_radius)

    def _generate_data(self, context):
        """Generate the differential photon counts. This is the main method of the data generation. The calculation
                procedure is as follows: For each time step, for each wavelength bin, for each photon source, get the sky
                brightness of the source and the intensity response vector at that time and then, for each differential output,
                calculate the differential photon counts.
                """
        differential_output_pairs = context.observatory.beam_combination_scheme.get_differential_output_pairs()
        time_range = context.time_range
        wavelength_bin_centers = context.observatory.instrument_parameters.wavelength_bin_centers
        target_specific_photon_sources = context.target_specific_photon_sources

        for index_time, source in product(range(len(time_range)), target_specific_photon_sources):

            time = time_range[index_time]
            time_copy = time
            if not context.settings.planet_orbital_motion:
                time_copy = 0 * u.s

            source_sky_coordinates = source.get_sky_coordinates(time_copy, context.settings.grid_size)
            source_sky_brightness_distribution_map = source.get_sky_brightness_distribution_map(time_copy,
                                                                                                source_sky_coordinates)

            for index_wavelength, wavelength in enumerate(wavelength_bin_centers):

                intensity_responses = self._get_intensity_responses(
                    time=time,
                    wavelength=wavelength,
                    source_sky_coordinates=source_sky_coordinates,
                    observatory_coordinates=context.observatory.array_configuration.get_collector_positions(time),
                    aperture_radius=context.observatory.instrument_parameters.aperture_radius,
                    beam_combination_matrix=context.observatory.beam_combination_scheme.get_beam_combination_transfer_matrix(),
                    number_of_inputs=context.observatory.beam_combination_scheme.number_of_inputs,
                    number_of_outputs=context.observatory.beam_combination_scheme.number_of_outputs,
                    fiber_injection_variability=context.settings.noise_contributions.fiber_injection_variability,
                    optical_path_difference_variability_apply=context.settings.noise_contributions.optical_path_difference_variability.apply,
                    optical_path_difference_distribution=context.settings.noise_contributions.optical_path_difference_distribution,
                    grid_size=context.settings.grid_size)

                photon_counts_per_output = self._get_photon_counts_per_output(
                    source_sky_brightness_distribution=source_sky_brightness_distribution_map,
                    wavelength_bin_width=context.observatory.instrument_parameters.wavelength_bin_widths[
                        index_wavelength],
                    index_wavelength=index_wavelength,
                    intensity_responses=intensity_responses,
                    time_step=context.settings.time_step,
                    unperturbed_instrument_throughput=context.observatory.instrument_parameters.unperturbed_instrument_throughput)

                for differential_output_pair in differential_output_pairs:
                    self.differential_photon_counts[index_wavelength][
                        index_time] += self._get_differential_photon_counts(
                        photon_counts_per_output=photon_counts_per_output,
                        differential_output_pair=differential_output_pair).value

                #     if self.animator and (
                #             source.name == self.animator.source_name and
                #             wavelength == self.animator.closest_wavelength and
                #             index_pair == self.animator.differential_intensity_response_index):
                #         self._update_animation_frame()

    def _get_differential_photon_counts(self, photon_counts_per_output, differential_output_pair):
        return photon_counts_per_output[differential_output_pair[0]] - photon_counts_per_output[
            differential_output_pair[1]]

    def _get_input_complex_amplitudes(self,
                                      time: astropy.units.Quantity,
                                      wavelength: astropy.units.Quantity,
                                      source_sky_coordinates: Coordinates,
                                      observatory_coordinates: Coordinates,
                                      aperture_radius: astropy.units.Quantity,
                                      number_of_inputs: int,
                                      grid_size: int) -> np.ndarray:
        """Return the input complex amplitude vector, consisting of a flat wavefront per collector.

        :param time: The time for which the intensity response is calculated
        :param wavelength: The wavelength for which the intensity response is calculated
        :param source_sky_coordinates: The sky coordinates of the source for which the intensity response is calculated
        :param observatory_coordinates: The observatory coordinates at the time
        :param aperture_radius: The aperture radius of the collectors
        :param number_of_inputs: The number of inputs, i.e. collectors
        :param grid_size: The grid size for the calculations
        :return: The input complex amplitude vector
        """
        input_complex_amplitudes = np.zeros((number_of_inputs, grid_size, grid_size), dtype=complex) * u.m

        for index_input in range(number_of_inputs):
            input_complex_amplitudes[index_input] = aperture_radius.to(u.m) * np.exp(1j * 2 * np.pi / wavelength * (
                    observatory_coordinates.x[index_input] * source_sky_coordinates.x.to(u.rad).value +
                    observatory_coordinates.y[index_input] * source_sky_coordinates.y.to(u.rad).value))
        return input_complex_amplitudes

    def _get_intensity_responses(self,
                                 time: astropy.units.Quantity,
                                 wavelength: astropy.units.Quantity,
                                 source_sky_coordinates: np.ndarray,
                                 observatory_coordinates: np.ndarray,
                                 aperture_radius: astropy.units.Quantity,
                                 beam_combination_matrix: np.ndarray,
                                 number_of_inputs: int,
                                 number_of_outputs: int,
                                 fiber_injection_variability: bool,
                                 optical_path_difference_variability_apply: bool,
                                 optical_path_difference_distribution: np.ndarray,
                                 grid_size: int) -> np.ndarray:
        """Return the intensity response vector consisting of the intensity responses of each input collector.

        :param time: The time
        :param wavelength: The wavelength
        :param source_sky_coordinates: The source sky coordinates
        :param observatory_coordinates: The observatory coordinates
        :param aperture_radius: The aperture radius
        :param beam_combination_matrix: The bea, combination transfer matrix
        :param number_of_inputs: The number of inputs, i.e. collectors
        :param number_of_outputs: The number of interferometric outputs
        :param fiber_injection_variability: Whether fiber injection variability should be modeled
        :param optical_path_difference_variability_apply: Whether optical path difference variability should be modeled
        :param optical_path_difference_distribution: The distribution to sample the OPD
        :param grid_size: The grid size for the calculations
        :return: The intensity response vector
        """
        input_complex_amplitudes = self._get_input_complex_amplitudes(time=time,
                                                                      wavelength=wavelength,
                                                                      source_sky_coordinates=source_sky_coordinates,
                                                                      observatory_coordinates=observatory_coordinates,
                                                                      aperture_radius=aperture_radius,
                                                                      number_of_inputs=number_of_inputs,
                                                                      grid_size=grid_size)
        input_complex_amplitudes = np.reshape(input_complex_amplitudes, (number_of_inputs, grid_size ** 2))

        if fiber_injection_variability or optical_path_difference_variability_apply:
            perturbation_matrix = self._get_perturbation_matrix(
                wavelength=wavelength,
                fiber_injection_variability=fiber_injection_variability,
                optical_path_difference_variability_apply=optical_path_difference_variability_apply,
                optical_path_difference_distribution=optical_path_difference_distribution,
                number_of_inputs=number_of_inputs)
            input_complex_amplitudes = np.dot(perturbation_matrix, input_complex_amplitudes)

        intensity_responses = abs(np.dot(beam_combination_matrix, input_complex_amplitudes)) ** 2
        intensity_responses = np.reshape(intensity_responses, (number_of_outputs, grid_size, grid_size))
        return intensity_responses

    def _get_perturbation_matrix(self,
                                 wavelength: astropy.units.Quantity,
                                 fiber_injection_variability: bool,
                                 optical_path_difference_variability_apply: bool,
                                 optical_path_difference_distribution: np.ndarray,
                                 number_of_inputs: int) -> np.ndarray:
        """Return the perturbation matrix with randomly generated noise.

        :param wavelength: The wavelength
        :param fiber_injection_variability: Whether fiber injection variability should be modeled
        :param optical_path_difference_variability_apply: Whether optical path difference variability should be modeled
        :param optical_path_difference_distribution: The distribution to sample the OPD
        :param number_of_inputs: The number of inputs, i.e. collectors
        :return:
        """
        diagonal_of_matrix = []

        for index in range(number_of_inputs):
            # TODO: Use more realistic distributions
            if fiber_injection_variability:
                amplitude_factor = np.random.uniform(0.8, 0.9)
            if optical_path_difference_variability_apply:
                phase_difference = choice(optical_path_difference_distribution).to(u.um)

            diagonal_of_matrix.append(amplitude_factor * np.exp(2j * np.pi / wavelength * phase_difference))

        return np.diag(diagonal_of_matrix)

    def _get_photon_counts_per_output(self,
                                      source_sky_brightness_distribution: np.ndarray,
                                      wavelength_bin_width: astropy.units.Quantity,
                                      index_wavelength: int,
                                      intensity_responses: np.ndarray,
                                      time_step: astropy.units.Quantity,
                                      unperturbed_instrument_throughput: float) -> list:
        """Return the photon counts per output, i.e. for a given intensity response map, including photon noise.

        :param mean_spectral_flux_density: Mean spectral density of the source at a given wavelength
        :param source_sky_brightness_distribution: Shape map of the source
        :param wavelength_bin_width: Wavelength bin width
        :param intensity_response: Intensity response map
        :return: Photon counts in units of photons
        """
        normalization = len(
            source_sky_brightness_distribution[index_wavelength][
                source_sky_brightness_distribution[index_wavelength].value > 0]) if not len(
            source_sky_brightness_distribution[index_wavelength][
                source_sky_brightness_distribution[index_wavelength].value > 0]) == 0 else 1

        photon_counts_per_output = []

        for intensity_response in intensity_responses:
            mean_photon_counts = (np.sum(intensity_response
                                         * source_sky_brightness_distribution[index_wavelength]
                                         * time_step.to(u.s)
                                         * wavelength_bin_width
                                         * unperturbed_instrument_throughput).value
                                  / normalization)

            photon_counts_per_output.append(self._get_photon_shot_noise(mean_photon_counts=mean_photon_counts))
        return photon_counts_per_output

    def _get_photon_shot_noise(self, mean_photon_counts: int):
        try:
            photon_counts = poisson(mean_photon_counts, 1)
        except ValueError:
            photon_counts = round(normal(mean_photon_counts, 1))
        return photon_counts * u.ph

    def _update_animation_frame(self, time, intensity_responses, pair_of_indices, index_pair, index_photon_source,
                                source, wavelength, index_time):
        self.animator.update_collector_position(time, self.observatory)
        self.animator.update_differential_intensity_response(
            intensity_responses[pair_of_indices[0]] - intensity_responses[pair_of_indices[1]])
        self.animator.update_differential_photon_counts(
            self.output[index_photon_source].differential_photon_counts_by_source[index_pair][
                source.name][
                wavelength][
                index_time], index_time)
        self.animator.writer.grab_frame()

    def apply(self, context: Context) -> Context:
        """Apply the modules.

        :param context: The contexts object of the pipelines
        :return: The (updated) contexts object
        """
        self._set_optimal_baseline(context)
        self.differential_photon_counts = np.zeros(
            (len(context.observatory.instrument_parameters.wavelength_bin_centers), len(context.time_range)))
        self._generate_data(context)

        return context
