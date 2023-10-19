import numpy as np


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
        self.photon_rate_time_series = dict((source.name, dict(
            (wavelength_bin_center,
             np.zeros((number_of_differential_intensity_responses, number_of_time_steps), dtype=float)) for
            wavelength_bin_center in self.wavelength_bin_centers)) for source in sources)

    @property
    def photon_rate_time_series_total(self):
        photon_rate_time_series_toal = dict(
            (wavelength_bin_center,
             np.zeros((self.number_of_differential_intensity_responses, self.number_of_time_steps), dtype=float)) for
            wavelength_bin_center in self.wavelength_bin_centers)
        for key_source in self.photon_rate_time_series.keys():
            for key_wavelength in self.photon_rate_time_series[key_source].keys():
                photon_rate_time_series_toal[key_wavelength] += self.photon_rate_time_series[key_source][key_wavelength]
        return photon_rate_time_series_toal
