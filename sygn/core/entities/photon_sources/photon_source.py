from abc import ABC, abstractmethod
from typing import Any

import astropy
import numpy as np
from pydantic import BaseModel

from sygn.util.helpers import Coordinates


class PhotonSource(ABC, BaseModel):
    """Class representation of a photon source.
    """

    # number_of_wavelength_bins: int
    # name: str
    # temperature: Any

    mean_spectral_flux_density: Any = None

    @abstractmethod
    def get_sky_coordinates(self, time: astropy.units.Quantity, grid_size: int) -> Coordinates:
        """Return the sky coordinate maps of the source. The intensity responses are calculated in a resolution that
        allows the source to fill the grid, thus, each source needs to define its own sky coordinate map.

        :param time: The time
        :param grid_size: The grid size
        :return: A tuple containing the x- and y-sky coordinate maps
        """
        pass

    @abstractmethod
    def get_sky_brightness_distribution_map(self, sky_coordinates: Coordinates) -> np.ndarray:
        """Return the sky brightness distribution map of the source object.

        :param sky_coordinates: The sky corodinates
        :return: The sky brightness distribution map
        """
        pass
