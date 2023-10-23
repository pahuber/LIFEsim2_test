from typing import Tuple

import astropy.units
import numpy as np
from astropy import units as u

from lifesim2.util.grid import get_index_of_closest_value


class SimulationOutput():
    """
    Class representation of the simulation outputs.
    """

    def __init__(self,
                 number_of_differential_intensity_responses: int,
                 number_of_time_steps: int,
                 wavelength_bin_centers: np.ndarray,
                 sources: np.ndarray):
        """Constructor method.

        :param number_of_differential_intensity_responses: The number of differential_intensity_responses
        :param number_of_time_steps: The number of time steps made in the simulation
        :param wavelength_bin_centers: The wavelength bin centers
        """
        self.number_of_differential_intensity_responses = number_of_differential_intensity_responses
        self.number_of_time_steps = number_of_time_steps
        self.wavelength_bin_centers = wavelength_bin_centers
        self.photon_rate_time_series = dict((sources[key].name, dict(
            (wavelength_bin_center,
             np.zeros((number_of_differential_intensity_responses, number_of_time_steps), dtype=float) * u.ph / u.s) for
            wavelength_bin_center in wavelength_bin_centers)) for key in sources.keys())
        self.photon_rate_time_series_total = dict((wavelength_bin_center, np.zeros(
            (number_of_differential_intensity_responses, number_of_time_steps), dtype=float) * u.ph / u.s) for
                                                  wavelength_bin_center
                                                  in wavelength_bin_centers)

    def _calculate_total_photon_rate_time_series(self):
        """Calculate the total photon rate time series by summing the photon rates of all sources.
        """
        for source in self.photon_rate_time_series.keys():
            for wavelength in self.photon_rate_time_series[source].keys():
                self.photon_rate_time_series_total[wavelength] += self.photon_rate_time_series[source][wavelength]

    def get_total_photon_rate_time_series(self,
                                          wavelength: astropy.units.Quantity,
                                          differential_intensity_response_index: int = 0) -> Tuple[
        np.ndarray, astropy.units.Quantity]:
        """Return the total photon rate time series for the wavelength that is closest to the given wavelength input.

        :param wavelength: Wavelength to return the time series for
        :param differential_intensity_response_index: Index describing for which of the potentially multiple
        differential intensity responses the photon rates should be returned
        :return: The photon rate time series for the given wavelength and the wavelength that has actually been used
        """
        index_of_closest_wavelength = get_index_of_closest_value(self.wavelength_bin_centers, wavelength)
        closest_wavelength = self.wavelength_bin_centers[index_of_closest_wavelength]
        return self.photon_rate_time_series_total[closest_wavelength][differential_intensity_response_index], np.round(
            closest_wavelength, 1)

    def get_photon_rate_time_series_for_source(self,
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
        return self.photon_rate_time_series[source_name][closest_wavelength][
            differential_intensity_response_index], np.round(closest_wavelength, 1)
