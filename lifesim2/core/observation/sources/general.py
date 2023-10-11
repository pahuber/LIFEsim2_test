import astropy
import numpy as np
from astropy import units as u

from lifesim2.core.observation.observation import Observation


def get_input_complex_amplitude_vector(observation: Observation, time: astropy.units.Quantity) -> np.ndarray:
    """Return the unperturbed input complex amplitude vector, consisting of a flat wavefront per collector.

    :param time: The time to calculate the vector at
    :return: The input complex amplitude vector
    """
    wavelength = 10 * u.um
    x_sky_coordinates = observation.x_sky_coordinates_map.to(u.rad).value
    y_sky_coordinates = observation.y_sky_coordinates_map.to(u.rad).value

    x_observatory_coordinates, y_observatory_coordinates = observation.observatory.array_configuration.get_collector_positions(
        time)

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
        diagonal_of_matrix.append(np.random.uniform(0.2, 0.4) * np.exp(1j * np.random.uniform(-0.2, 0.2)))

    perturbation_matrix = np.diag(diagonal_of_matrix)
    # perturbation_matrix = np.diag([1, 1, 1])

    return perturbation_matrix
