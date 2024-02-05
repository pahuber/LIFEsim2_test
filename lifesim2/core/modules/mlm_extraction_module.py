from itertools import product
from typing import Tuple

import numpy as np
from astropy import units as u

from lifesim2.core.context import Context
from lifesim2.core.extraction import Extraction
from lifesim2.core.modules.base_module import BaseModule
from lifesim2.core.modules.data_generator_module import DataGeneratorModule
from lifesim2.core.modules.fits_reader_module import FITSReaderModule
from lifesim2.core.modules.template_generator_module import TemplateGeneratorModule
from lifesim2.util.grid import get_indices_of_maximum_of_2d_array
from lifesim2.util.helpers import FITSReadWriteType


class MLExtractionModule(BaseModule):
    """Class representation of the maximum likelihood extraction module.
    """

    def __init__(self):
        """Constructor method.
        """
        self.dependencies = [(FITSReaderModule, FITSReadWriteType.SyntheticMeasurement, FITSReadWriteType.Template),
                             (FITSReaderModule, DataGeneratorModule, FITSReadWriteType.Template),
                             (FITSReaderModule, TemplateGeneratorModule, FITSReadWriteType.SyntheticMeasurement),
                             (DataGeneratorModule, TemplateGeneratorModule)]

    def _calculate_maximum_likelihood(self, signal, context) -> Tuple:
        """Calculate the maximum likelihood estimate for the flux in units of photons at the position of the maximum of
        the cost function.

        :param signal: The signal
        :param context: The context object of the pipeline
        :return: The cost function and the optimum flux
        """

        cost_function = np.zeros((context.observatory.beam_combination_scheme.number_of_differential_outputs,
                                  context.settings.grid_size, context.settings.grid_size,
                                  len(context.observatory.instrument_parameters.wavelength_bin_centers)))
        optimum_flux = np.zeros(cost_function.shape)

        for index_x, index_y in product(range(context.settings.grid_size), range(context.settings.grid_size)):

            template = context.templates[index_x, index_y]

            matrix_c = self._get_matrix_c(signal, template.signal)
            matrix_b = self._get_matrix_b(signal, template.signal)

            for index_output in range(len(matrix_b)):
                optimum_flux[index_output, index_x, index_y] = self._get_optimum_flux(matrix_b[index_output],
                                                                                      matrix_c[index_output])

                optimum_flux[index_output, index_x, index_y] = self._get_positivity_constraint(
                    optimum_flux[index_output, index_x, index_y])

                # Calculate the cost function according to equation B.8
                cost_function[index_output, index_x, index_y] = (optimum_flux[index_output, index_x, index_y] *
                                                                 matrix_c[index_output])

        # Sum cost function over all wavelengths
        cost_function = np.sum(cost_function, axis=3)
        cost_function[np.isnan(cost_function)] = 0
        return cost_function, optimum_flux

    def _get_fluxes_uncertainties(self, cost_functions, cost_functions_white, optimum_fluxes_white,
                                  context) -> np.ndarray:
        """Return the uncertainties on the extracted fluxes by calculating the standard deviation of the extracted
        fluxes at positions around the center at a radius of the maximum cost function.

        :param cost_functions: The cost functions
        :param cost_functions_white: The whitened cost functions
        :param optimum_fluxes_white: The whitened optimum fluxes
        :param context: The context object of the pipeline
        :return: The uncertainties on the extracted fluxes
        """
        for index_output in range(len(cost_functions_white)):
            # Get extracted flux at positions around center at radius of maximum cost function
            height, width = cost_functions_white[index_output, :, :].shape
            index_x, index_y = get_indices_of_maximum_of_2d_array(cost_functions[index_output])

            # Create a boolean mask for the circle
            center = (width // 2, height // 2)
            radius = np.sqrt((index_x - width // 2) ** 2 + (index_y - width // 2) ** 2)
            y, x = np.ogrid[:height, :width]
            mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2 + 0.5) & \
                   ((x - center[0]) ** 2 + (y - center[1]) ** 2 >= (radius - 1) ** 2 + 0.5)
            # TODO: Remove planet position from mask

            # # Extract the pixels within the circle
            # plt.imshow(cost_functions_white[0, :, :] * mask)
            # plt.colorbar()
            # plt.show()

            masked_fluxes_white_flattened = np.einsum('ijk, ij -> ijk', optimum_fluxes_white[index_output, :, :],
                                                      mask).reshape(
                context.settings.grid_size ** 2, -1)

            uncertainties = np.zeros(masked_fluxes_white_flattened.shape[1], dtype=object)

            for index in range(masked_fluxes_white_flattened.shape[1]):
                # Remove zero values from array, since they are not part of the mask and would change the standard
                # deviation
                non_zero_values = [el for el in masked_fluxes_white_flattened[:, index] if el > 0]
                uncertainties[index] = np.std(non_zero_values)

        return uncertainties

    def _get_matrix_c(self, signal: np.ndarray, template_signal: np.ndarray) -> np.ndarray:
        """Calculate the matrix C according to equation B.2.

        :param signal: The signal
        :param template_signal: The template signal
        :return: The matrix C
        """
        data_variance = np.var(signal, axis=2)
        return np.sum(signal * template_signal, axis=2) / data_variance

    def _get_matrix_b(self, signal: np.ndarray, template_signal: np.ndarray) -> np.ndarray:
        """Calculate the matrix B according to equation B.3.

        :param signal: The signal
        :param template_signal: The template signal
        :return: The matrix B
        """
        data_variance = np.var(signal, axis=2)
        matrix_b_elements = np.sum(template_signal ** 2, axis=2) / data_variance
        matrix_b = np.zeros(matrix_b_elements.shape[0], dtype=object)

        for index_output in range(len(matrix_b_elements)):
            matrix_b[index_output] = np.diag(matrix_b_elements[index_output])

        return matrix_b

    def _get_optimum_flux(self, matrix_b: np.ndarray, matrix_c: np.ndarray) -> np.ndarray:
        """Calculate the optimum flux according to equation B.6.

        :param matrix_b: The matrix B
        :param matrix_c: The matrix C
        :return: The optimum flux
        """
        return np.diag(np.linalg.inv(matrix_b) * matrix_c)

    def _get_optimum_flux_at_cost_function_maximum(self, cost_functions, optimum_fluxes, context) -> np.ndarray:
        """Calculate the optimum flux at the position of the maximum of the cost function.

        :param cost_functions: The cost functions
        :param optimum_fluxes: The optimum fluxes
        :param context: The context object of the pipeline
        :return: The optimum flux at the position of the maximum of the cost function
        """
        optimum_flux_at_maximum = np.zeros(
            (context.observatory.beam_combination_scheme.number_of_differential_outputs,
             len(context.observatory.instrument_parameters.wavelength_bin_centers))) * u.ph

        for index_output in range(len(optimum_flux_at_maximum)):
            index_x, index_y = get_indices_of_maximum_of_2d_array(cost_functions[index_output])
            optimum_flux_at_maximum[index_output] = optimum_fluxes[index_output, index_x, index_y] * u.ph

        return optimum_flux_at_maximum

    def _get_positivity_constraint(self, optimum_flux: np.ndarray) -> np.ndarray:
        """Return the optimum flux with negative values set to zero.

        :param optimum_flux: The optimum flux
        :return: The optimum flux with negative values set to zero
        """
        return np.where(optimum_flux >= 0, optimum_flux, 0)

    def _get_whitened_signal(self, signal, optimum_fluxes, cost_functions, context) -> np.ndarray:
        """Return the whitened signal, i.e. the original signal with the most likely planet signal substracted.

        :param signal: The signal
        :param optimum_fluxes: The optimum fluxes
        :param cost_functions: The cost functions
        :param context: The context object of the pipeline
        :return: The whitened signal
        """
        for index_output in range(context.observatory.beam_combination_scheme.number_of_differential_outputs):
            index_x, index_y = get_indices_of_maximum_of_2d_array(cost_functions[index_output])
            signal_white = np.copy(context.signal)
            signal_white -= np.einsum('ij, ijk->ijk', optimum_fluxes[:, index_x, index_y],
                                      context.templates[index_x, index_y].signal)
        return signal_white

    def apply(self, context: Context) -> Context:
        """Calculate the function for each template and then get a maximum likelihood estimate for the flux in units of
        photons at the position of the maximum of the cost function.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        cost_functions, optimum_fluxes = self._calculate_maximum_likelihood(context.signal, context)
        optimum_flux_at_maximum = self._get_optimum_flux_at_cost_function_maximum(cost_functions,
                                                                                  optimum_fluxes,
                                                                                  context)

        # Get the whitened signal and the uncertainties on the extracted flux
        signal_white = self._get_whitened_signal(context.signal, optimum_fluxes, cost_functions, context)
        cost_functions_white, optimum_fluxes_white = self._calculate_maximum_likelihood(signal_white, context)
        optimum_fluxes_uncertainties = self._get_fluxes_uncertainties(cost_functions,
                                                                      cost_functions_white,
                                                                      optimum_fluxes_white,
                                                                      context)

        context.extractions.append(Extraction(optimum_flux_at_maximum, optimum_fluxes_uncertainties, cost_functions))

        return context
