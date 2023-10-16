from typing import Tuple

import astropy.units
import numpy as np


def get_meshgrid(full_extent: astropy.units.Quantity, grid_size: int) -> Tuple[
    np.ndarray, np.ndarray]:
    """Return a tuple of numpy arrays corresponding to a meshgrid.

    :param full_extent: Full extent in one dimension
    :param grid_size: Grid size
    :return: Tuple of numpy arrays
    """
    linspace = np.linspace(-full_extent.value / 2, full_extent.value / 2, grid_size)
    return np.meshgrid(linspace, linspace) * full_extent.unit


def get_sky_coordinates(wavelengths, field_of_view, grid_size):
    x_sky_coordinates = np.zeros((len(wavelengths), grid_size, grid_size))
    y_sky_coordinates = np.zeros((len(wavelengths), grid_size, grid_size))
    for index, wavelength in enumerate(wavelengths):
        x_sky_coordinates, y_sky_coordinates = get_meshgrid(field_of_view[index], grid_size)
    return x_sky_coordinates, y_sky_coordinates
