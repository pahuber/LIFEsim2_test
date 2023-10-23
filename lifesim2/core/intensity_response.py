import astropy
import numpy as np
from astropy import units as u

from lifesim2.core.noise_contributions import NoiseContributions
from lifesim2.core.observatory.observatory import Observatory


def get_differential_intensity_responses(time,
                                         wavelength,
                                         observatory: Observatory,
                                         source_sky_coordinate_maps: np.ndarray,
                                         grid_size: int,
                                         noise_contributions: NoiseContributions) -> np.ndarray:
    """Return an array containing the differential intensity responses (differential virtual transmission maps), given an intensity response
     vector. For certain beam combination schemes, multiple differential intensity responses exist.

    :param time: The time for which the intensity response is calculated
    :param wavelength: The wavelength for which the intensity response is calculated
    :param observatory: The observatory for which the intensity response is calculated
    :param source_sky_coordinate_maps: The sky coordinates of the source for which the intensity response is calculated
    :param grid_size: The grid size
    :param noise_contributions: The noise contributions object
    :return: An array containing the differential intensity responses
    """
    intensity_response_vector = get_intensity_responses(time, wavelength, observatory, source_sky_coordinate_maps,
                                                        grid_size, noise_contributions)
    differential_indices = observatory.beam_combination_scheme.get_differential_intensity_response_indices()
    differential_intensity_responses = np.zeros(
        (len(differential_indices), grid_size,
         grid_size)) * intensity_response_vector.unit
    for index_index, index_pair in enumerate(differential_indices):
        differential_intensity_responses[index_index] = intensity_response_vector[index_pair[0]] - \
                                                        intensity_response_vector[
                                                            index_pair[1]]
    return differential_intensity_responses


def get_intensity_responses(time: astropy.units.Quantity,
                            wavelength: astropy.units.Quantity,
                            observatory: Observatory,
                            source_sky_coordinate_maps,
                            grid_size: int,
                            noise_contributions: NoiseContributions) -> np.ndarray:
    """Return the intensity response vector.

    :param time: The time for which the intensity response is calculated
    :param wavelength: The wavelength for which the intensity response is calculated
    :param observatory: The observatory for which the intensity response is calculated
    :param source_sky_coordinate_maps: The sky coordinates of the source for which the intensity response is calculated
    :param grid_size: The grid size
    :param noise_contributions: The noise contributions object
    :return: The intensity response vector
    """
    input_complex_amplitude_unperturbed_vector = np.reshape(
        get_input_complex_amplitude_vector(time, wavelength, observatory, source_sky_coordinate_maps,
                                           grid_size), (
            observatory.beam_combination_scheme.number_of_inputs, grid_size ** 2))

    perturbation_matrix = get_perturbation_matrix(observatory, noise_contributions)

    input_complex_amplitude_perturbed_vector = np.dot(perturbation_matrix,
                                                      input_complex_amplitude_unperturbed_vector)

    beam_combination_matrix = observatory.beam_combination_scheme.get_beam_combination_transfer_matrix()

    intensity_response_perturbed_vector = np.reshape(abs(
        np.dot(beam_combination_matrix, input_complex_amplitude_perturbed_vector)) ** 2,
                                                     (
                                                         observatory.beam_combination_scheme.number_of_outputs,
                                                         grid_size,
                                                         grid_size))

    return intensity_response_perturbed_vector


def get_input_complex_amplitude_vector(time: astropy.units.Quantity,
                                       wavelength: astropy.units.Quantity,
                                       observatory: Observatory,
                                       source_sky_coordinate_maps,
                                       grid_size: int) -> np.ndarray:
    """Return the unperturbed input complex amplitude vector, consisting of a flat wavefront per collector.

    :param time: The time for which the intensity response is calculated
    :param wavelength: The wavelength for which the intensity response is calculated
    :param observatory: The observatory for which the intensity response is calculated
    :param source_sky_coordinate_maps: The sky coordinates of the source for which the intensity response is calculated
    :param grid_size: The grid size
    :return: The input complex amplitude vector
    """
    x_observatory_coordinates, y_observatory_coordinates = observatory.array_configuration.get_collector_positions(
        time)
    input_complex_amplitude_vector = np.zeros((observatory.beam_combination_scheme.number_of_inputs,
                                               grid_size, grid_size),
                                              dtype=complex) * observatory.instrument_parameters.aperture_radius.unit

    for index_input in range(observatory.beam_combination_scheme.number_of_inputs):
        input_complex_amplitude_vector[index_input] = observatory.instrument_parameters.aperture_radius * np.exp(
            1j * 2 * np.pi / wavelength * (
                    x_observatory_coordinates[index_input] * source_sky_coordinate_maps[0].to(u.rad).value +
                    y_observatory_coordinates[index_input] * source_sky_coordinate_maps[1].to(u.rad).value))
    return input_complex_amplitude_vector


def get_perturbation_matrix(observatory: Observatory, noise_contributions: NoiseContributions) -> np.ndarray:
    """Return the perturbation matrix with randomly generated noise.

    :param observatory: The observatory for which the intensity response is calculated
    :param noise_contributions: The noise contributions object
    :return: The perturbation matrix
    """
    if (noise_contributions.fiber_injection_variability
            or noise_contributions.optical_path_difference_variability):

        diagonal_of_matrix = []

        for index in range(observatory.beam_combination_scheme.number_of_inputs):
            amplitude_factor = 1
            phase_difference = 0

            if noise_contributions.fiber_injection_variability:
                amplitude_factor = np.random.uniform(0.4, 0.6)
            if noise_contributions.optical_path_difference_variability:
                phase_difference = np.random.uniform(-0.0000000001, 0.0000000001)

            diagonal_of_matrix.append(amplitude_factor * np.exp(1j * phase_difference))

        return np.diag(diagonal_of_matrix)

    return np.diag(np.ones(observatory.beam_combination_scheme.number_of_inputs))
