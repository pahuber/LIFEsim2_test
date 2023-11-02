from typing import Tuple

import astropy
import numpy as np
from astropy import units as u

from lifesim2.util.grid import get_index_of_closest_value


class SyntheticData():
    def __init__(self,
                 sources: np.ndarray,
                 wavelength_bin_centers: np.ndarray,
                 number_of_differential_intensity_responses: int,
                 number_of_time_steps: int):
        self.wavelength_bin_centers = wavelength_bin_centers
        self.photon_count_time_series_by_source = dict((sources[key].name, dict((wavelength_bin_center, np.zeros(
            (number_of_differential_intensity_responses, number_of_time_steps), dtype=float) * u.ph) for
                                                                                wavelength_bin_center in
                                                                                wavelength_bin_centers)) for key in
                                                       sources.keys())
        self.photon_count_time_series = dict((wavelength_bin_center, np.zeros(
            (number_of_differential_intensity_responses, number_of_time_steps), dtype=float) * u.ph) for
                                             wavelength_bin_center
                                             in wavelength_bin_centers)

    def get_total_photon_count_time_series(self,
                                           wavelength: astropy.units.Quantity,
                                           differential_intensity_response_index: int = 0) -> Tuple[
        np.ndarray, astropy.units.Quantity]:
        """Return the total photon count time series for the wavelength that is closest to the given wavelength input.

        :param wavelength: Wavelength to return the time series for
        :param differential_intensity_response_index: Index describing for which of the potentially multiple
        differential intensity responses the photon rates should be returned
        :return: The photon count time series for the given wavelength and the wavelength that has actually been used
        """
        index_of_closest_wavelength = get_index_of_closest_value(self.wavelength_bin_centers, wavelength)
        closest_wavelength = self.wavelength_bin_centers[index_of_closest_wavelength]
        return self.photon_count_time_series[closest_wavelength][
            differential_intensity_response_index], np.round(
            closest_wavelength, 1)

    def get_photon_count_time_series_for_source(self,
                                                source_name: str,
                                                wavelength: astropy.units.Quantity,
                                                differential_intensity_response_index: int = 0) -> Tuple[
        np.ndarray, astropy.units.Quantity]:
        """Return the photon rate time series for a specific source for the wavelength that is closest to the given wavelength input.

        :param source_name: Name of the source
        :param wavelength: Wavelength to return the time series for
        :param differential_intensity_response_index: Index describing for which of the potentially multiple
        differential intensity responses the photon rates should be returned
        :return: The photon rate time series for the given wavelength and the wavelength that has actually been used
        """
        index_of_closest_wavelength = get_index_of_closest_value(self.wavelength_bin_centers, wavelength)
        closest_wavelength = self.wavelength_bin_centers[index_of_closest_wavelength]
        return self.photon_count_time_series_by_source[source_name][closest_wavelength][
            differential_intensity_response_index], np.round(closest_wavelength, 1)
