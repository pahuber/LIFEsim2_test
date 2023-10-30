from typing import Tuple

import astropy.units
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from lifesim2.util.grid import get_index_of_closest_value


class SimulationOutput():
    """
    Class representation of the simulation outputs.
    """

    def __init__(self,
                 number_of_differential_intensity_responses: int,
                 number_of_time_steps: int,
                 wavelength_bin_centers: np.ndarray,
                 sources: np.ndarray,
                 time_range: np.ndarray):
        """Constructor method.

        :param number_of_differential_intensity_responses: The number of differential_intensity_responses
        :param number_of_time_steps: The number of time steps made in the simulation
        :param wavelength_bin_centers: The wavelength bin centers
        """
        self.number_of_differential_intensity_responses = number_of_differential_intensity_responses
        self.number_of_time_steps = number_of_time_steps
        self.wavelength_bin_centers = wavelength_bin_centers
        self.sources = sources
        self.time_range = time_range
        self.photon_count_time_series = None
        self.photon_count_time_series_total = dict((wavelength_bin_center, np.zeros(
            (number_of_differential_intensity_responses, number_of_time_steps), dtype=float) * u.ph) for
                                                   wavelength_bin_center
                                                   in wavelength_bin_centers)
        self.photon_counts_per_wavelength_bin = dict((sources[key].name, dict(
            (wavelength_bin_center,
             np.zeros(len(wavelength_bin_centers), dtype=int) * u.ph) for
            wavelength_bin_center in wavelength_bin_centers)) for key
                                                     in sources.keys())

    def _calculate_photon_counts_per_wavelength_bin(self):
        for source_name in self.sources.keys():
            for wavelength in self.wavelength_bin_centers:
                self.photon_counts_per_wavelength_bin[source_name][wavelength] = np.sum(np.abs(
                    self.photon_count_time_series[source_name][wavelength]))

    def _calculate_total_photon_count_time_series(self):
        """Calculate the total photon count time series by summing the photon rates of all sources.
        """
        for source in self.photon_count_time_series.keys():
            for wavelength in self.photon_count_time_series[source].keys():
                self.photon_count_time_series_total[wavelength] += self.photon_count_time_series[source][wavelength]

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
        return self.photon_count_time_series_total[closest_wavelength][
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
        return self.photon_count_time_series[source_name][closest_wavelength][
            differential_intensity_response_index], np.round(closest_wavelength, 1)

    def plot_photon_count_time_series(self,
                                      source_name: str,
                                      wavelength: astropy.units.Quantity,
                                      differential_intensity_response_index: int = 0):
        """Plot the photon count time series for the total signal and for an additional source at a given wavelength.

        :param source_name: Name of the source
        :param wavelength: Wavelength to return the time series for
        :param differential_intensity_response_index: Index describing for which of the potentially multiple
        differential intensity responses the photon rates should be returned
        """
        photon_rate_time_series_total, closest_wavelength = self.get_total_photon_count_time_series(wavelength,
                                                                                                    differential_intensity_response_index)
        labels = (int(label.to(u.h).value) for label in self.time_range[::10])
        plt.plot(photon_rate_time_series_total, 'b-o', label=f'Total')
        plt.plot(self.get_photon_count_time_series_for_source(source_name, wavelength,
                                                              differential_intensity_response_index)[0],
                 'r-o',
                 label=f'{source_name} at {closest_wavelength}')
        plt.title('Photon Count Time Series')
        plt.ylabel('Photon Counts')
        plt.xlabel('Time (h)')
        plt.xticks(ticks=range(len(self.time_range))[::10], labels=labels)
        plt.legend()
        plt.tight_layout()
        plt.show()
