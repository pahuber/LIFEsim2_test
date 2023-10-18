import astropy.units
import numpy as np

from lifesim2.core.sources.source import Source


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
                           x_sky_coordinates,
                           y_sky_coordinates,
                           source: Source):
        """Append the photon rates for the different sources to the time series. Done for each time step.

        :param time_index: The index of the current time step
        :param transmission_maps: An array containing the transmission maps
        """
        for index, differential_intensity_response in enumerate(differential_intensity_responses):
            index_x = np.abs(x_sky_coordinates[0, :] - source.position[0]).argmin()
            index_y = np.argmin(np.abs(y_sky_coordinates[:, 0] - source.position[1]))
            self.photon_rate_time_series[wavelength][index][time_index] = \
                differential_intensity_responses[index][index_x][index_y] * source.flux[wavelength_index].value
