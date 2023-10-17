import astropy.units
import numpy as np


class SimulationOutput():
    """
    Class representation of the simulation outputs.
    """

    def __init__(self, number_of_differential_intensity_responses: int, number_of_time_steps: int,
                 wavelength_bin_centers: np.ndarray):
        """Constructor method.

        :param number_of_differential_intensity_responses: The number of differential_intensity_responses
        :param number_of_time_steps: The number of time steps made in the simulation
        """
        self.wavelength_bin_centers = wavelength_bin_centers
        self.photon_rate_time_series = dict(
            (wavelength_bin_center, np.zeros((number_of_differential_intensity_responses, number_of_time_steps))) for
            wavelength_bin_center in self.wavelength_bin_centers)

    def append_photon_rate(self,
                           time_index: int,
                           differential_intensity_responses: np.ndarray,
                           wavelength: astropy.units.Quantity,
                           wavelength_index,
                           source_flux: np.ndarray):
        """Append the photon rates for the different sources to the time series. Done for each time step.

        :param time_index: The index of the current time step
        :param transmission_maps: An array containing the transmission maps
        """
        for index, differential_intensity_response in enumerate(differential_intensity_responses):
            self.photon_rate_time_series[wavelength][index][time_index] = \
                differential_intensity_responses[index][100 // 4][100 // 4] * source_flux[wavelength_index].value
