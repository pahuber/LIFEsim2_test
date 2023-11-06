from typing import Tuple, Any

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


def get_number_of_instances_in_list(list: list, instance_type: Any) -> int:
    """Return the number of objects of a given instance in a list.

    :param list: The list
    :param instance_type: The type of instance
    :return: The number of objects
    """
    return len([value for value in list if isinstance(value, instance_type)])


def get_dictionary_from_list_containing_key(key: str, list: list) -> dict:
    """Given a list of dictionaries and a key, return the dictionary that contains the key.

    :param key: The key
    :param list: The list
    :return: The dictionary containing the key
    """
    for dictionary in list:
        if key in dictionary:
            return dictionary
