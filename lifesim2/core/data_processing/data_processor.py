import math
from enum import Enum

import astropy
import matplotlib
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from lifesim2.core.simulation.simulation import Simulation


class DataProcessingMode(Enum):
    PHOTON_STATISTICS = 1
    EXTRACTION = 2


class DataProcessor():
    def __init__(self, simulation: Simulation):
        self.simulation = simulation

    def plot_photon_count_time_series(self,
                                      source_names: list[str],
                                      wavelength: astropy.units.Quantity,
                                      plot_total_counts: bool = True,
                                      time_units: astropy.units.Quantity = u.h,
                                      differential_intensity_response_index: int = 0):
        """Plot the photon count time series for the total signal and for an additional source at a given wavelength.

        :param source_names: Names of the sources to plot
        :param wavelength: Wavelength to return the time series for
        :param plot_total_counts: Whether or notthe total photon counts should be plotted as well
        :param differential_intensity_response_index: Index describing for which of the potentially multiple
        differential intensity responses the photon rates should be returned
        """
        photon_rate_time_series_total, closest_wavelength = self.simulation.output.get_total_photon_count_time_series(
            wavelength,
            differential_intensity_response_index)
        labels = (np.round(label.to(time_units).value, 1) for label in self.simulation.config.time_range[::10])
        matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=["r", "k", "c"])
        if plot_total_counts:
            plt.plot(photon_rate_time_series_total, 'b-o', label=f'Total')
        for source_name in source_names:
            plt.plot(self.simulation.output.get_photon_count_time_series_for_source(source_name, wavelength,
                                                                                    differential_intensity_response_index)[
                         0],
                     '-o',
                     label=f'{source_name} at {closest_wavelength}')
        plt.title('Photon Count Time Series')
        plt.ylabel('Photon Counts')
        plt.xlabel(f'Time ({str(time_units)})')
        plt.xticks(ticks=range(len(self.simulation.config.time_range))[::10], labels=labels)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_total_signal_to_noise_ratio_vs_time(self):
        snrs = []

        for index_time in range(len(self.simulation.config.time_range)):
            if index_time > 1:
                snr_squared = 0
                for wavelength in self.simulation.observation.observatory.instrument_parameters.wavelength_bin_centers:
                    photon_counts_planet = \
                        self.simulation.output.get_photon_count_time_series_for_source('Earth', wavelength)[0]
                    photon_counts_total = self.simulation.output.get_total_photon_count_time_series(wavelength)[0]
                    signal = np.sum(abs(photon_counts_planet.value)[:index_time])
                    noise = np.sqrt(np.sum(abs(photon_counts_total.value)[:index_time]))
                    if not (math.isnan(signal) or math.isnan(noise) or noise == 0):
                        snr_squared += (signal / noise) ** 2
                snrs.append(np.sqrt(snr_squared))

        def fit(x, a):
            return a * np.sqrt(x)

        popt, _ = curve_fit(fit, range(len(snrs)), snrs)
        # popt_lin, _ = curve_fit(fit_linear, range(len(snrs)), snrs)

        plt.plot(snrs)
        plt.plot(fit(range(len(snrs)), popt[0]))
        # plt.plot(fit_linear(range(len(snrs)), popt_lin[0], popt_lin[1]))
        plt.show()
        # print(popt_lin)

    def get_snr_per_bin(self, mode: DataProcessingMode):
        pass

    def get_total_snr(self, mode: DataProcessingMode):
        pass

    def get_recovered_spectrum(self, mode: DataProcessingMode):
        pass
