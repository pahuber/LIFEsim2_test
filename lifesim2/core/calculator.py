import astropy
import numpy as np
from astropy import units as u

from lifesim2.core.observation import Observation


def get_differential_intensity_responses(time,
                                         wavelength,
                                         observation: Observation,
                                         grid_size: int) -> np.ndarray:
    """Return the transmission map(s), given an intensity response vector. For certain bea, combination schemes,
    multiple transmission maps exist.

    :param intensity_response_vector: The intensity response vector
    :param beam_combination_scheme: The beam combination scheme
    :param grid_size: The grid size for the calculations
    :return: An array containing the transmission map(s)
    """
    intensity_response_vector = get_intensity_responses(time, wavelength, observation, grid_size)
    indices = observation.observatory.beam_combination_scheme.get_differential_intensity_response_indices()
    transmission_maps = np.zeros((len(indices), grid_size, grid_size))
    for index_index, index_pair in enumerate(indices):
        transmission_maps[index_index] = intensity_response_vector[index_pair[0]] - intensity_response_vector[
            index_pair[1]]
    return transmission_maps


def get_intensity_responses(time: astropy.units.Quantity,
                            wavelength: astropy.units.Quantity,
                            observation: Observation,
                            grid_size: int) -> np.ndarray:
    """Return the intensity response vector.

    :param time: Time to calculate the intensity responese vector for
    :param observation: The observation object
    :param beam_combination_matrix: The beam combination matrix
    :param grid_size: The grid size for the calculations
    :return: The intensity response vector
    """
    input_complex_amplitude_unperturbed_vector = np.reshape(
        get_input_complex_amplitude_vector(observation, time, wavelength), (
            observation.observatory.beam_combination_scheme.number_of_inputs, grid_size ** 2))

    perturbation_matrix = get_perturbation_matrix(observation)

    input_complex_amplitude_perturbed_vector = np.dot(perturbation_matrix,
                                                      input_complex_amplitude_unperturbed_vector)

    beam_combination_matrix = observation.observatory.beam_combination_scheme.get_beam_combination_transfer_matrix()
    intensity_response_vector = np.reshape(abs(
        np.dot(beam_combination_matrix, input_complex_amplitude_perturbed_vector)) ** 2,
                                           (
                                               observation.observatory.beam_combination_scheme.number_of_outputs,
                                               grid_size,
                                               grid_size))
    return intensity_response_vector


def get_input_complex_amplitude_vector(observation: Observation, time: astropy.units.Quantity,
                                       wavelength: astropy.units.Quantity) -> np.ndarray:
    """Return the unperturbed input complex amplitude vector, consisting of a flat wavefront per collector.

    :param time: The time to calculate the vector at
    :return: The input complex amplitude vector
    """
    x_sky_coordinates = observation.x_sky_coordinates_map.to(u.rad).value
    y_sky_coordinates = observation.y_sky_coordinates_map.to(u.rad).value

    x_observatory_coordinates, y_observatory_coordinates = observation.observatory.array_configuration.get_collector_positions(
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

    input_complex_amplitude_vector = np.zeros((observation.observatory.beam_combination_scheme.number_of_inputs,
                                               observation.grid_size, observation.grid_size),
                                              dtype=complex)

    for index_input in range(observation.observatory.beam_combination_scheme.number_of_inputs):
        input_complex_amplitude_vector[index_input] = (
                observation.observatory.instrument_parameters.aperture_radius * np.exp(
            1j * 2 * np.pi / wavelength * (
                    x_observatory_coordinates[index_input] * x_sky_coordinates + y_observatory_coordinates[
                index_input] * y_sky_coordinates)))

    return input_complex_amplitude_vector


def get_perturbation_matrix(observation: Observation) -> np.ndarray:
    """Return the perturbation matrix with randomly generated noise.

    :return: The perturbation matrix
    """
    diagonal_of_matrix = []
    for index in range(observation.observatory.beam_combination_scheme.number_of_inputs):
        diagonal_of_matrix.append(np.random.uniform(0.4, 0.6) * np.exp(1j * np.random.uniform(-0.2, 0.2)))

    perturbation_matrix = np.diag(diagonal_of_matrix)
    # perturbation_matrix = np.diag([1, 1, 1])

    return perturbation_matrix
