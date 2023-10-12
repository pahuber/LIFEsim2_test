import numpy as np


class SimulationOutput():
    """
    Class representation of the simulation outputs.
    """

    def __init__(self, number_of_transmission_maps: int, number_of_time_steps: int):
        """Constructor method.

        :param number_of_transmission_maps: The number of transmission maps
        :param number_of_time_steps: The number of time steps made in the simulation
        """
        self.photon_rate_time_series = np.zeros((number_of_transmission_maps, number_of_time_steps))

    def append_photon_rate(self, time_index: int, transmission_maps: np.ndarray):
        """Append the photon rates for the different sources to the time series. Done for each time step.

        :param time_index: The index of the current time step
        :param transmission_maps: An array containing the transmission maps
        """
        for index, transmission_map in enumerate(transmission_maps):
            self.photon_rate_time_series[index][time_index] = transmission_maps[index][100 // 4][
                100 // 4]
