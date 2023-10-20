import astropy
import numpy as np
from astropy import units as u

from lifesim2.core.observatory.observatory import Observatory


def get_differential_intensity_responses(time,
                                         wavelength,
                                         observatory: Observatory,
                                         source_sky_coordinate_maps: np.ndarray,
                                         grid_size: int) -> np.ndarray:
    """Return an array containing the differential intensity responses (differential virtual transmission maps), given an intensity response
     vector. For certain beam combination schemes, multiple differential intensity responses exist.

    :param time: The time for which the intensity response is calculated
    :param wavelength: The wavelength for which the intensity response is calculated
    :param observatory: The observatory for which the intensity response is calculated
    :param source_sky_coordinate_maps: The sky coordinates of the source for which the intensity response is calculated
    :param grid_size: The grid size
    :return: An array containing the differential intensity responses
    """
    # TODO: fix units
    intensity_response_vector = get_intensity_responses(time, wavelength, observatory, source_sky_coordinate_maps,
                                                        grid_size)
    differential_indices = observatory.beam_combination_scheme.get_differential_intensity_response_indices()
    differential_intensity_responses = np.zeros((len(differential_indices), grid_size, grid_size))
    for index_index, index_pair in enumerate(differential_indices):
        differential_intensity_responses[index_index] = intensity_response_vector[index_pair[0]] - \
                                                        intensity_response_vector[
                                                            index_pair[1]]
    return differential_intensity_responses


def get_intensity_responses(time: astropy.units.Quantity,
                            wavelength: astropy.units.Quantity,
                            observatory: Observatory,
                            source_sky_coordinate_maps,
                            grid_size: int) -> np.ndarray:
    """Return the intensity response vector.

    :param time: The time for which the intensity response is calculated
    :param wavelength: The wavelength for which the intensity response is calculated
    :param observatory: The observatory for which the intensity response is calculated
    :param source_sky_coordinate_maps: The sky coordinates of the source for which the intensity response is calculated
    :param grid_size: The grid size
    :return: The intensity response vector
    """
    input_complex_amplitude_unperturbed_vector = np.reshape(
        get_input_complex_amplitude_vector(time, wavelength, observatory, source_sky_coordinate_maps, grid_size), (
            observatory.beam_combination_scheme.number_of_inputs, grid_size ** 2))

    perturbation_matrix = get_perturbation_matrix(observatory)

    input_complex_amplitude_perturbed_vector = np.dot(perturbation_matrix,
                                                      input_complex_amplitude_unperturbed_vector)

    beam_combination_matrix = observatory.beam_combination_scheme.get_beam_combination_transfer_matrix()

    intensity_response_unperturbed_vector = np.reshape(abs(
        np.dot(beam_combination_matrix, input_complex_amplitude_unperturbed_vector)) ** 2,
                                                       (
                                                           observatory.beam_combination_scheme.number_of_outputs,
                                                           grid_size,
                                                           grid_size))

    intensity_response_perturbed_vector = np.reshape(abs(
        np.dot(beam_combination_matrix, input_complex_amplitude_perturbed_vector)) ** 2,
                                                     (
                                                         observatory.beam_combination_scheme.number_of_outputs,
                                                         grid_size,
                                                         grid_size))

    # Normalize perturbed intensity response vector
    for index in range(len(intensity_response_perturbed_vector)):
        intensity_response_perturbed_vector[index] /= np.max(intensity_response_unperturbed_vector[index])
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

    # plt.plot(x_observatory_coordinates[0].value, y_observatory_coordinates[0].value, 'bo')
    # plt.plot(x_observatory_coordinates[1].value, y_observatory_coordinates[1].value, 'bo')
    # plt.plot(x_observatory_coordinates[2].value, y_observatory_coordinates[2].value, 'bo')
    # # plt.plot(x_observatory_coordinates[3].value, y_observatory_coordinates[3].value, 'bo')
    # plt.gca().set_aspect('equal')
    # plt.xlim(-40, 40)
    # plt.ylim(-40, 40)
    # plt.savefig(f'array_{time}.png')
    # plt.close()

    input_complex_amplitude_vector = np.zeros((observatory.beam_combination_scheme.number_of_inputs,
                                               grid_size, grid_size),
                                              dtype=complex)

    for index_input in range(observatory.beam_combination_scheme.number_of_inputs):
        input_complex_amplitude_vector[index_input] = (observatory.instrument_parameters.aperture_radius * np.exp(
            1j * 2 * np.pi / wavelength * (
                    x_observatory_coordinates[index_input] * source_sky_coordinate_maps[0].to(u.rad).value +
                    y_observatory_coordinates[index_input] * source_sky_coordinate_maps[1].to(u.rad).value)))
    return input_complex_amplitude_vector


def get_perturbation_matrix(observatory: Observatory) -> np.ndarray:
    """Return the perturbation matrix with randomly generated noise.

    :return: The perturbation matrix
    """
    diagonal_of_matrix = []
    for index in range(observatory.beam_combination_scheme.number_of_inputs):
        diagonal_of_matrix.append(
            np.random.uniform(0.6, 0.8) * np.exp(1j * np.random.uniform(-0.001, 0.001)))

    perturbation_matrix = np.diag(diagonal_of_matrix)
    # perturbation_matrix = np.diag([1, 1, 1, 1])

    return perturbation_matrix
