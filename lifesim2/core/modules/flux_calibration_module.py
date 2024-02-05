import numpy as np
from astropy import units as u

from lifesim2.core.context import Context
from lifesim2.core.modules.base_module import BaseModule
from lifesim2.core.modules.mlm_extraction_module import MLExtractionModule
from lifesim2.util.grid import get_indices_of_maximum_of_2d_array


class FluxCalibrationModule(BaseModule):
    """Class representation of the flux calibration module.
    """

    def __init__(self):
        """Constructor method.
        """
        self.dependencies = [(MLExtractionModule,)]

    def apply(self, context: Context) -> Context:
        """For every extraction in the context, calibrate the flux and convert it from photon counts to spectral flux
        density by dividing by the time step, wavelength bin widths and effective areas.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        for index_extraction, extraction in enumerate(context.extractions):
            flux_density = np.zeros(extraction.spectrum.shape) * u.ph / (
                    u.m ** 2 * u.um * u.s)
            flux_density_uncertainty = np.zeros(extraction.spectrum.shape) * u.ph / (
                    u.m ** 2 * u.um * u.s)

            for index_differential_output in range(len(extraction.spectrum)):
                index_x, index_y = get_indices_of_maximum_of_2d_array(
                    extraction.cost_function[index_differential_output])

                effective_area = context.templates[index_x, index_y].effective_area_rms
                time_step = context.settings.time_step.to(u.s)
                wavelength_bin_widths = context.observatory.instrument_parameters.wavelength_bin_widths

                flux_density[index_differential_output] = (
                        extraction.spectrum[index_differential_output]
                        / time_step
                        / wavelength_bin_widths
                        / effective_area)

                flux_density_uncertainty[index_differential_output] = (
                        extraction.spectrum_uncertainties[index_differential_output] * u.ph
                        / time_step
                        / wavelength_bin_widths
                        / effective_area)

            context.extractions[index_extraction].spectrum = flux_density
            context.extractions[index_extraction].spectrum_uncertainties = flux_density_uncertainty
        return context
