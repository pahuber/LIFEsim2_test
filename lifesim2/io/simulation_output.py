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
            (wavelength_bin_center,
             np.zeros((number_of_differential_intensity_responses, number_of_time_steps), dtype=float)) for
            wavelength_bin_center in self.wavelength_bin_centers)
