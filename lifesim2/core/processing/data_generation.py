from enum import Enum
from itertools import product
from typing import Tuple

import astropy
import numpy as np
from astropy import units as u
from numpy.random import poisson, normal
from tqdm import tqdm

from lifesim2.core.context import Context
from lifesim2.util.helpers import Coordinates


class GenerationMode(Enum):
    """Class representing the data generation mode.
    """
    data = 0
    template = 1


class DataGenerator():
    """Class representing the data generator.
    """

    def __init__(self, context: Context, mode: GenerationMode):
        """The constructor method.

        :param context: The context
        :param mode: The data generation mode
        """
        self._context = context
        self._mode = mode
        self.differential_photon_counts = np.zeros(
            (self._context.observatory.beam_combination_scheme.number_of_differential_outputs,
             len(self._context.observatory.instrument_parameters.wavelength_bin_centers),
             len(self._context.time_range)))
        self.differential_effective_area = np.zeros(
            (self._context.observatory.beam_combination_scheme.number_of_differential_outputs,
             len(self._context.observatory.instrument_parameters.wavelength_bin_centers),
             len(self._context.time_range))) * u.m ** 2

    def _get_differential_photon_counts(self, photon_counts_per_output, differential_output_pair) -> np.ndarray:
        """Return the differential photon counts, given the photon counts per output and the pair of outputs.

        :param photon_counts_per_output: The photon counts per output
        :param differential_output_pair: The differential output pair
        :return: The differential photon counts
        """
        return photon_counts_per_output[differential_output_pair[0]] - photon_counts_per_output[
            differential_output_pair[1]]

    def _get_index_planet_motion(self, index_time: int) -> int:
        """Return the time index that is used by the methods that depend on the planet orbital motion. If the planet
        orbital motion is not considered, the time index is 0, otherwise it is identical to the main time index.

        :param index_time: The time index
        :return: The time index considering the planet orbital motion
        """
        if self._context.settings.planet_orbital_motion:
            return index_time
        return 0

    def _get_input_complex_amplitudes(self,
                                      time: astropy.units.Quantity,
                                      time_index: int,
                                      wavelength: astropy.units.Quantity,
                                      source_sky_coordinates: Coordinates,
                                      observatory_coordinates: Coordinates,
                                      aperture_radius: astropy.units.Quantity,
                                      number_of_inputs: int,
                                      amplitude_perturbation_apply: bool,
                                      phase_perturbation_apply: bool,
                                      phase_perturbation_distribution: np.ndarray,
                                      polarization_perturbation_apply: bool,
                                      polarization_perturbation_distribution: np.ndarray,
                                      grid_size: int) -> np.ndarray:
        """Return the input complex amplitude vector, consisting of a flat wavefront per collector.

        :param time: The time for which the intensity response is calculated
        :param time_index: The time index
        :param wavelength: The wavelength for which the intensity response is calculated
        :param source_sky_coordinates: The sky coordinates of the source for which the intensity response is calculated
        :param observatory_coordinates: The observatory coordinates at the time
        :param aperture_radius: The aperture radius of the collectors
        :param number_of_inputs: The number of inputs, i.e. collectors
        :param amplitude_perturbation_apply: Whether fiber injection variability should be modeled
        :param phase_perturbation_apply: Whether optical path difference variability should be modeled
        :param phase_perturbation_distribution: The distribution to sample the OPD
        :param grid_size: The grid size for the calculations
        :return: The input complex amplitude vector
        """
        input_complex_amplitudes = np.zeros((number_of_inputs, 2, grid_size, grid_size), dtype=complex) * u.m

        for index_input in range(number_of_inputs):
            amplitude_perturbation = 1
            phase_perturbation = 0 * u.um
            polarization_perturbation = 0 * u.rad
            # TODO: check this value = 0 is ok
            polarization_angle = 0 * u.rad  # We can set this to zero here without loss of generality
            if amplitude_perturbation_apply:
                amplitude_perturbation = np.random.uniform(0.8, 0.9)
            if phase_perturbation_apply:
                phase_perturbation = (phase_perturbation_distribution[index_input][time_index]).to(u.um)
            if polarization_perturbation_apply:
                polarization_perturbation = (polarization_perturbation_distribution[index_input][time_index]).to(u.rad)

            # TODO: Add eta total here instead of photon counts calculation
            input_complex_amplitudes[index_input][0] = amplitude_perturbation * aperture_radius.to(u.m) * np.exp(
                1j * 2 * np.pi / wavelength * (
                        observatory_coordinates.x[index_input] * source_sky_coordinates.x.to(u.rad).value +
                        observatory_coordinates.y[index_input] * source_sky_coordinates.y.to(
                    u.rad).value + phase_perturbation)) * np.cos(polarization_angle + polarization_perturbation)

            input_complex_amplitudes[index_input][1] = amplitude_perturbation * aperture_radius.to(u.m) * np.exp(
                1j * 2 * np.pi / wavelength * (
                        observatory_coordinates.x[index_input] * source_sky_coordinates.x.to(u.rad).value +
                        observatory_coordinates.y[index_input] * source_sky_coordinates.y.to(
                    u.rad).value + phase_perturbation)) * np.sin(polarization_angle + polarization_perturbation)
        return input_complex_amplitudes

    def _get_intensity_responses(self,
                                 time: astropy.units.Quantity,
                                 time_index: int,
                                 wavelength: astropy.units.Quantity,
                                 source_sky_coordinates: np.ndarray,
                                 observatory_coordinates: np.ndarray,
                                 aperture_radius: astropy.units.Quantity,
                                 beam_combination_matrix: np.ndarray,
                                 number_of_inputs: int,
                                 number_of_outputs: int,
                                 amplitude_perturbation_apply: bool,
                                 phase_perturbation_apply: bool,
                                 phase_perturbation_distribution: np.ndarray,
                                 polarization_perturbation_apply: bool,
                                 polarization_perturbation_distribution: np.ndarray,
                                 grid_size: int) -> np.ndarray:
        """Return the intensity response vector consisting of the intensity responses of each input collector.

        :param time: The time
        :param time_index: The time index
        :param wavelength: The wavelength
        :param source_sky_coordinates: The source sky coordinates
        :param observatory_coordinates: The observatory coordinates
        :param aperture_radius: The aperture radius
        :param beam_combination_matrix: The bea, combination transfer matrix
        :param number_of_inputs: The number of inputs, i.e. collectors
        :param number_of_outputs: The number of interferometric outputs
        :param amplitude_perturbation_apply: Whether fiber injection variability should be modeled
        :param phase_perturbation_apply: Whether optical path difference variability should be modeled
        :param phase_perturbation_distribution: The distribution to sample the OPD
        :param polarization_perturbation_apply: Whether polarization perturbations should be modeled
        :param polarization_perturbation_distribution: The distribution to sample the polarization perturbations
        :param grid_size: The grid size for the calculations
        :return: The intensity response vector
        """
        input_complex_amplitudes = self._get_input_complex_amplitudes(time=time,
                                                                      time_index=time_index,
                                                                      wavelength=wavelength,
                                                                      source_sky_coordinates=source_sky_coordinates,
                                                                      observatory_coordinates=observatory_coordinates,
                                                                      aperture_radius=aperture_radius,
                                                                      number_of_inputs=number_of_inputs,
                                                                      amplitude_perturbation_apply=amplitude_perturbation_apply,
                                                                      phase_perturbation_apply=phase_perturbation_apply,
                                                                      phase_perturbation_distribution=phase_perturbation_distribution,
                                                                      polarization_perturbation_apply=polarization_perturbation_apply,
                                                                      polarization_perturbation_distribution=polarization_perturbation_distribution,
                                                                      grid_size=grid_size)
        input_complex_amplitudes = np.reshape(input_complex_amplitudes, (number_of_inputs, 2, grid_size ** 2))

        # if amplitude_perturbation_apply or phase_perturbation_apply:
        #     perturbation_matrix = self._get_perturbation_matrix(
        #         time_index=time_index,
        #         wavelength=wavelength,
        #         fiber_injection_variability=amplitude_perturbation_apply,
        #         phase_perturbations_apply=phase_perturbation_apply,
        #         phase_perturbation_distribution=phase_perturbation_distribution,
        #         number_of_inputs=number_of_inputs)
        #     input_complex_amplitudes = np.dot(perturbation_matrix, input_complex_amplitudes)

        intensity_responses = abs(np.dot(beam_combination_matrix, input_complex_amplitudes[:, 0])) ** 2 + abs(
            np.dot(beam_combination_matrix, input_complex_amplitudes[:, 1])) ** 2
        intensity_responses = np.reshape(intensity_responses, (number_of_outputs, grid_size, grid_size))
        return intensity_responses

    def _get_normalization(self, source_sky_brightness_distribution: np.ndarray) -> float:
        """Return the normalization that accounts for the discretization of the sky brightness distribution map into
        pixels. Count all pixels that have a non-zero value.

        :param source_sky_brightness_distribution: The source sky brightness distribution
        :return: The normalization
        """
        return len(source_sky_brightness_distribution[source_sky_brightness_distribution.value > 0]) if not len(
            source_sky_brightness_distribution[source_sky_brightness_distribution.value > 0]) == 0 else 1

    # def _get_perturbation_matrix(self,
    #                              time_index: int,
    #                              wavelength: astropy.units.Quantity,
    #                              fiber_injection_variability: bool,
    #                              phase_perturbations_apply: bool,
    #                              phase_perturbation_distribution: np.ndarray,
    #                              number_of_inputs: int) -> np.ndarray:
    #     """Return the perturbation matrix with randomly generated noise.
    #
    #     :param time_index: The time index
    #     :param wavelength: The wavelength
    #     :param fiber_injection_variability: Whether fiber injection variability should be modeled
    #     :param phase_perturbations_apply: Whether phase perturbations sould be modeled
    #     :param phase_perturbation_distribution: The distribution to sample the OPD
    #     :param number_of_inputs: The number of inputs, i.e. collectors
    #     :return:
    #     """
    #     diagonal_of_matrix = []
    #
    #     for index in range(number_of_inputs):
    #         amplitude_factor = 1
    #         phase_difference = 0 * u.um
    #         # TODO: Use more realistic distributions
    #         if fiber_injection_variability:
    #             amplitude_factor = np.random.uniform(0.8, 0.9)
    #         if phase_perturbations_apply:
    #             phase_difference = (phase_perturbation_distribution[index][time_index]).to(u.um)
    #
    #         diagonal_of_matrix.append(amplitude_factor * np.exp(2j * np.pi / wavelength * phase_difference))
    #
    #     return np.diag(diagonal_of_matrix)

    def _get_photon_counts_per_output(self,
                                      source_sky_brightness_distribution: np.ndarray,
                                      wavelength_bin_width: astropy.units.Quantity,
                                      index_wavelength: int,
                                      intensity_responses: np.ndarray,
                                      time_step: astropy.units.Quantity,
                                      unperturbed_instrument_throughput: float) -> Tuple[list, np.ndarray]:
        """Return the photon counts per output, i.e. for a given intensity response map, including photon noise.

        :param mean_spectral_flux_density: Mean spectral density of the source at a given wavelength
        :param source_sky_brightness_distribution: Shape map of the source
        :param wavelength_bin_width: Wavelength bin width
        :param intensity_response: Intensity response map
        :return: Photon counts in units of photons
        """
        normalization = self._get_normalization(source_sky_brightness_distribution)
        photon_counts_per_output = []

        if self._mode == GenerationMode.template:
            effective_area = np.zeros(len(intensity_responses)) * u.m ** 2

            # Normalize planet sky brightness distribution to 1
            source_sky_brightness_distribution[np.isnan(source_sky_brightness_distribution)] = 0
            source_sky_brightness_distribution /= source_sky_brightness_distribution.value.max()

            for index_intensity_response, intensity_response in enumerate(intensity_responses):
                mean_photon_counts = (np.sum(intensity_response
                                             * source_sky_brightness_distribution
                                             * time_step.to(u.s)
                                             * wavelength_bin_width
                                             * unperturbed_instrument_throughput).value
                                      / normalization)

                # Calculate effective area, i.e. area including all throughput terms, using the pixel of the intensity
                # response where the planet is located
                effective_area[index_intensity_response] = np.sum(
                    intensity_response * unperturbed_instrument_throughput * source_sky_brightness_distribution.value)
                photon_counts_per_output.append(mean_photon_counts * u.ph)
            return photon_counts_per_output, effective_area

        else:
            for index_intensity_response, intensity_response in enumerate(intensity_responses):
                mean_photon_counts = (np.sum(intensity_response
                                             * source_sky_brightness_distribution
                                             * time_step.to(u.s)
                                             * wavelength_bin_width
                                             * unperturbed_instrument_throughput).value
                                      / normalization)

                photon_counts_per_output.append(self._get_photon_shot_noise(mean_photon_counts=mean_photon_counts))
            return photon_counts_per_output, None

    def _get_photon_shot_noise(self, mean_photon_counts: int) -> astropy.units.Quantity:
        """Given an amount of photons, calculate and return the amount of photons given by drawing from a Poisson/
        Gaussian distribution.

        :param mean_photon_counts: The mean photon counts
        :return: The photon counts considering shot noise
        """
        try:
            photon_counts = poisson(mean_photon_counts, 1)
        except ValueError:
            photon_counts = round(normal(mean_photon_counts, 1))
        return photon_counts * u.ph

    def _update_animation_frame(self, time, intensity_responses, pair_of_indices, index_pair, index_wavelength,
                                source, wavelength, index_time):
        self._context.animator.update_collector_position(time, self._context.observatory)
        self._context.animator.update_differential_intensity_response(
            intensity_responses[pair_of_indices[0]] - intensity_responses[pair_of_indices[1]])
        self._context.animator.update_differential_photon_counts(
            self.differential_photon_counts[index_pair][index_wavelength][
                index_time], index_time)
        self._context.animator.writer.grab_frame()

    def generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate the differential photon counts. This is the main method of the data generation. The calculation
        procedure is as follows: For each time step, for each wavelength bin, for each photon source, calculate  the
        intensity response vector at that time, wavelength and angular resolution corresponding to the source and then,
        for each differential output, calculate the differential photon counts.
        """
        for index_time, time in enumerate(
                tqdm(self._context.time_range, disable=self._mode == GenerationMode.template)):

            index_time_planet_motion = self._get_index_planet_motion(index_time)

            for index_wavelength, source in product(
                    range(len(self._context.observatory.instrument_parameters.wavelength_bin_centers)),
                    self._context.photon_sources
            ):

                wavelength = self._context.observatory.instrument_parameters.wavelength_bin_centers[index_wavelength]

                # Calculate the vector of intensity responses, each intensity response corresponding to one output
                intensity_responses = self._get_intensity_responses(
                    time=time,
                    time_index=index_time,
                    wavelength=wavelength,
                    source_sky_coordinates=source.get_sky_coordinates(index_time_planet_motion, index_wavelength),
                    observatory_coordinates=self._context.observatory.array_configuration.get_collector_positions(time),
                    aperture_radius=self._context.observatory.instrument_parameters.aperture_radius,
                    beam_combination_matrix=self._context.observatory.beam_combination_scheme.get_beam_combination_transfer_matrix(),
                    number_of_inputs=self._context.observatory.beam_combination_scheme.number_of_inputs,
                    number_of_outputs=self._context.observatory.beam_combination_scheme.number_of_outputs,
                    amplitude_perturbation_apply=self._context.settings.noise.amplitude_perturbations,
                    phase_perturbation_apply=self._context.settings.noise.phase_perturbations.apply,
                    phase_perturbation_distribution=self._context.settings.noise.phase_perturbation_distribution,
                    polarization_perturbation_apply=self._context.settings.noise.polarization_perturbations.apply,
                    polarization_perturbation_distribution=self._context.settings.noise.polarization_perturbation_distribution,
                    grid_size=self._context.settings.grid_size)

                # Calculate the photon counts at each of the outputs
                photon_counts_per_output, effective_area = self._get_photon_counts_per_output(
                    source_sky_brightness_distribution=source.get_sky_brightness_distribution(index_time_planet_motion,
                                                                                              index_wavelength),
                    wavelength_bin_width=self._context.observatory.instrument_parameters.wavelength_bin_widths[
                        index_wavelength],
                    index_wavelength=index_wavelength,
                    intensity_responses=intensity_responses,
                    time_step=self._context.settings.time_step,
                    unperturbed_instrument_throughput=self._context.observatory.instrument_parameters.unperturbed_instrument_throughput)

                # For each pair of differential outputs, calculate the differential photon counts
                for index_pair, differential_output_pair in enumerate(
                        self._context.observatory.beam_combination_scheme.get_differential_output_pairs()):
                    self.differential_photon_counts[index_pair][index_wavelength][
                        index_time] += self._get_differential_photon_counts(
                        photon_counts_per_output=photon_counts_per_output,
                        differential_output_pair=differential_output_pair).value
                    if self._mode == GenerationMode.template:
                        self.differential_effective_area[index_pair][index_wavelength][index_time] = effective_area[
                                                                                                         differential_output_pair[
                                                                                                             0]] - \
                                                                                                     effective_area[
                                                                                                         differential_output_pair[
                                                                                                             1]]

                    if self._context.animator and (
                            source.name == self._context.animator.planet_name and
                            wavelength == self._context.animator.closest_wavelength and
                            index_pair == self._context.animator.differential_intensity_response_index):
                        self._update_animation_frame(time, intensity_responses, differential_output_pair, index_pair,
                                                     index_wavelength, source, wavelength, index_time)
        return self.differential_photon_counts, self.differential_effective_area
