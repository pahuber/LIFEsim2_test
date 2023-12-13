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
        flux_density = np.zeros(context.optimized_flux.shape) * u.ph / (u.m ** 2 * u.um * u.s)
        effective_area = \
            np.array(context.effective_area_rms).reshape(
                (context.settings.grid_size, context.settings.grid_size,
                 context.observatory.beam_combination_scheme.number_of_differential_outputs,
                 len(context.observatory.instrument_parameters.wavelength_bin_widths))) * u.m ** 2
        for index_output in range(len(context.optimized_flux)):
            j = context.cost_function[:, :, index_output]
            j_max = np.max(j)
            index = np.where(j == j_max)
            i1, i2 = index[0][0], index[1][0]

            time_step = context.settings.time_step.to(u.s)
            # time_step = context.mission.integration_time.to(u.s)
            wavelength_bin_widths = context.observatory.instrument_parameters.wavelength_bin_widths
            flux_density[index_output] = (context.optimized_flux[
                                              index_output] / time_step / wavelength_bin_widths /  # u.m ** 2)
                                          effective_area[i1, i2, index_output])
        context.flux_density = flux_density
        return context
