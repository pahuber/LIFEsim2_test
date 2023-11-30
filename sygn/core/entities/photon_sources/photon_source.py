from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel

from sygn.core.context import Context
from sygn.util.helpers import Coordinates


class PhotonSource(ABC, BaseModel):
    """Class representation of a photon source.
    """

    mean_spectral_flux_density: Any = None
    sky_brightness_distribution: Any = None
    sky_coordinates: Any = None

    @abstractmethod
    def _calculate_mean_spectral_flux_density(self, context: Context) -> np.ndarray:
        """Return the mean spectral flux density of the source object.

        :return: The mean spectral flux density
        """
        pass

    @abstractmethod
    def _calculate_sky_brightness_distribution(self, context: Context) -> np.ndarray:
        """Return the sky brightness distribution map of the source object.

        :return: The sky brightness distribution map
        """
        pass

    @abstractmethod
    def _calculate_sky_coordinates(self, context: Context) -> np.ndarray:
        """Return the sky coordinate maps of the source. The intensity responses are calculated in a resolution that
        allows the source to fill the grid, thus, each source needs to define its own sky coordinate map.

        :return: A tuple containing the x- and y-sky coordinate maps
        """
        pass

    @abstractmethod
    def get_sky_brightness_distribution(self, index_time: int, index_wavelength: int) -> np.ndarray:
        pass

    @abstractmethod
    def get_sky_coordinates(self, index_time: int, index_wavelength: int) -> Coordinates:
        pass

    def setup(self, context: Context):
        """Set up the main source properties. Rather than calling this method on initialization of the class instance,
        it has to be called explicitly after initiating the source. This ensures a flexibility in adapting source
        properties (e.g. temperature) on the fly without having to load a separate configuration file for each
        adaptation.
        """
        self.mean_spectral_flux_density = self._calculate_mean_spectral_flux_density(context)
        self.sky_coordinates = self._calculate_sky_coordinates(context)
        self.sky_brightness_distribution = self._calculate_sky_brightness_distribution(context)
