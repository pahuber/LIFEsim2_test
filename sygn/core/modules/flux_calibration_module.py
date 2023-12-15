import numpy as np
from astropy import units as u

from sygn.core.context import Context
from sygn.core.modules.base_module import BaseModule
from sygn.core.modules.mlm_extraction_module import MLExtractionModule


class FluxCalibrationModule(BaseModule):
    """Class representation of the flux calibration module.
    """

    def __init__(self):
        """Constructor method.
        """
        self.dependencies = [(MLExtractionModule,)]

    def apply(self, context: Context) -> Context:
        """Apply the module.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        for index_extraction, extraction in enumerate(context.extractions):
            flux_density = np.zeros(extraction.spectrum.shape) * u.ph / (
                    u.m ** 2 * u.um * u.s)
            flux_density_u = np.zeros(extraction.spectrum.shape) * u.ph / (
                    u.m ** 2 * u.um * u.s)

            for index_output in range(len(extraction.spectrum)):
                j = extraction.cost_function[index_output]
                j_max = np.max(j)
                index = np.where(j == j_max)
                i1, i2 = index[0][0], index[1][0]

                effective_area = context.templates[i1, i2].effective_area_rms
                time_step = context.settings.time_step.to(u.s)
                wavelength_bin_widths = context.observatory.instrument_parameters.wavelength_bin_widths

                flux_density[index_output] = (
                        extraction.spectrum[index_output] / time_step / wavelength_bin_widths / effective_area)
                flux_density_u[index_output] = (extraction.spectrum_uncertainties[
                                                    index_output] * u.ph / time_step / wavelength_bin_widths / effective_area)

            context.extractions[index_extraction].spectrum = flux_density
            context.extractions[index_extraction].spectrum_uncertainties = flux_density_u
        return context
