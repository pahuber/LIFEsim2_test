import numpy as np
from astropy import units as u
from scipy.optimize import curve_fit

from sygn.core.context import Context
from sygn.core.modules.base_module import BaseModule
from sygn.util.blackbody import create_blackbody_spectrum


class SpectrumFittingModule(BaseModule):
    """Class representation of the spectrum fitting module.
    """

    def __init__(self):
        """Constructor method.
        """
        self.dependencies = []

    def apply(self, context: Context) -> Context:
        """Fit a blackbody spectrum to the data.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        for extraction in context.extractions:
            # popt, pcov = curve_fit(lambda lam, T, radius: get_blackbody_spectrum(lam, T, radius,
            #                                                                      star_distance=context.star.distance.to(
            #                                                                          u.m).value),
            #                        context.observatory.instrument_parameters.wavelength_bin_centers.to(u.m).value,
            #                        extraction.spectrum.value[0], p0=[300, 7e6])
            popt, pcov = curve_fit(lambda lam, temp, radius: create_blackbody_spectrum(temperature=temp,
                                                                                       wavelength_bin_centers=lam,
                                                                                       source_solid_angle=None,
                                                                                       wavelength_range_lower_limit=context.observatory.instrument_parameters.wavelength_range_lower_limit.value,
                                                                                       wavelength_range_upper_limit=context.observatory.instrument_parameters.wavelength_range_upper_limit.value,
                                                                                       wavelength_bin_widths=context.observatory.instrument_parameters.wavelength_bin_widths.value,
                                                                                       is_fitting_mode=True,
                                                                                       planet_radius=radius,
                                                                                       star_distance=context.star.distance.to(
                                                                                           u.m).value),
                                   context.observatory.instrument_parameters.wavelength_bin_centers.value,
                                   extraction.spectrum.value[0], p0=[300, 7e6])
            radius = popt[1]
            solid_angle = np.pi * (radius / context.star.distance.to(u.m).value) ** 2 * u.sr

            extraction.spectrum_fit = create_blackbody_spectrum(temperature=popt[0] * u.K,
                                                                wavelength_bin_centers=context.observatory.instrument_parameters.wavelength_bin_centers,
                                                                wavelength_range_lower_limit=context.observatory.instrument_parameters.wavelength_range_lower_limit,
                                                                wavelength_range_upper_limit=context.observatory.instrument_parameters.wavelength_range_upper_limit,
                                                                wavelength_bin_widths=context.observatory.instrument_parameters.wavelength_bin_widths,
                                                                source_solid_angle=solid_angle)
            # extraction.spectrum_fit_uncertainties = np.sqrt(np.diag(pcov))

            print(popt, pcov)

        return context
