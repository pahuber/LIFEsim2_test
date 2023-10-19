from typing import Tuple

import astropy.units
import numpy as np


def get_meshgrid(full_extent: astropy.units.Quantity, grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return a tuple of numpy arrays corresponding to a meshgrid.

    :param full_extent: Full extent in one dimension
    :param grid_size: Grid size
    :return: Tuple of numpy arrays
    """
    linspace = np.linspace(-full_extent.value / 2, full_extent.value / 2, grid_size)
    return np.meshgrid(linspace, linspace) * full_extent.unit


def get_index_of_closest_value(array: np.ndarray, value: astropy.units.Quantity):
    """Return the index of a value in an array closest to the provided value.

    :param array: The array to search in
    :param value: The value to check
    :return: The index of the closest value
    """
    return np.abs(array - value).argmin()
