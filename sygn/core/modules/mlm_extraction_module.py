from itertools import product

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from sygn.core.context import Context
from sygn.core.extraction import Extraction
from sygn.core.modules.base_module import BaseModule
from sygn.core.modules.data_generator_module import DataGeneratorModule
from sygn.core.modules.fits_reader_module import FITSReaderModule
from sygn.core.modules.template_generator_module import TemplateGeneratorModule
from sygn.util.helpers import FITSReadWriteType


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

    def apply(self, context: Context) -> Context:
        """Apply the module.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        # Create cost function and extracted spectrum arrays of shape (number differential outputs, grid size, grid size,
        # number wavelengths)
        cost_function = np.zeros((context.observatory.beam_combination_scheme.number_of_differential_outputs,
                                  context.settings.grid_size, context.settings.grid_size,
                                  len(context.observatory.instrument_parameters.wavelength_bin_centers)))
        cost_function_white = np.zeros(cost_function.shape)
        extracted_spectrum = np.zeros(cost_function.shape)
        extracted_spectrum_white = np.zeros(cost_function.shape)

        data_variance = np.var(context.signal, axis=2)

        for index_x, index_y in product(range(context.settings.grid_size),
                                        range(context.settings.grid_size)):
            # for index_template, template in enumerate(context.templates):
            template = context.templates[index_x, index_y]
            # For each template and all outputs, calculate the matrix c and the elements of matrix B, according to
            # equations B.2 and B.3
            matrix_c = np.sum(context.signal * template.signal, axis=2) / data_variance
            matrix_b_elements = np.sum(template.signal ** 2, axis=2) / data_variance
            matrix_b = np.zeros(matrix_b_elements.shape[0], dtype=object)

            for index_output in range(len(matrix_b_elements)):
                # For each output, create the diagonal matrix B
                matrix_b[index_output] = np.diag(matrix_b_elements[index_output])

                # Calculate estimated flux vector according to equation B.6
                extracted_spectrum[index_output, index_x, index_y] = np.diag(
                    np.linalg.inv(matrix_b[index_output]) * matrix_c[
                        index_output])

                # Set positivity constraint
                extracted_spectrum[index_output, index_x, index_y] = np.where(
                    extracted_spectrum[index_output, index_x, index_y] >= 0,
                    extracted_spectrum[index_output, index_x, index_y], 0)

                # Calculate the cost function according to equation B.8
                cost_function[index_output, index_x, index_y] = extracted_spectrum[index_output, index_x, index_y] * \
                                                                matrix_c[index_output]

        # Sum cost function over all wavelengths
        cost_function = np.sum(cost_function, axis=3)

        # Get the estimated spectrum at the cost function maximum
        estimated_spectrum_cost_function_maximum = np.zeros(
            (context.observatory.beam_combination_scheme.number_of_differential_outputs,
             len(context.observatory.instrument_parameters.wavelength_bin_centers))) * u.ph

        for index_output in range(len(estimated_spectrum_cost_function_maximum)):
            j = cost_function[index_output, :, :]

            # Get extracted flux at position of maximum cost function
            j[np.isnan(j)] = 0
            j_max = np.max(j)
            index = np.where(j == j_max)
            i1, i2 = index[0][0], index[1][0]
            estimated_spectrum_cost_function_maximum[index_output] = extracted_spectrum[index_output, i1, i2] * u.ph

            # White data, i.e remove best fit template from time series

            # print(templates[i1, i2, index_output].shape)
            # print(context.data.differential_photon_count_time_series[index_output].shape)
            photon_time_series = context.signal
            # print(extracted_photon_counts.shape)
            # print(templates[i1, i2].shape)
            # print(photon_time_series.shape)
            photon_time_series -= np.einsum('ij, ijk->ijk', extracted_spectrum[:, i1, i2],
                                            context.templates[i1, i2].signal)

        # Repeat everything from above with whitened photon time series
        data_variance = np.var(photon_time_series, axis=2)

        for index_x, index_y in product(range(context.settings.grid_size),
                                        range(context.settings.grid_size)):
            # for index_template, template in enumerate(context.templates):
            template = context.templates[index_x, index_y]

            # For each template and all outputs, calculate the matrix c and the elements of matrix B, according to
            # equations B.2 and B.3
            matrix_c = np.sum(photon_time_series * template.signal, axis=2) / data_variance
            matrix_b_elements = np.sum(template.signal ** 2, axis=2) / data_variance
            matrix_b = np.zeros(matrix_b_elements.shape[0], dtype=object)

            for index_output in range(len(matrix_b_elements)):
                # For each output, create the diagonal matrix B
                matrix_b[index_output] = np.diag(matrix_b_elements[index_output])

                # Calculate estimated flux vector according to equation B.6
                extracted_spectrum_white[index_output, index_x, index_y] = np.diag(
                    np.linalg.inv(matrix_b[index_output]) * matrix_c[
                        index_output])

                # Set positivity constraint
                extracted_spectrum_white[index_output, index_x, index_y] = np.where(
                    extracted_spectrum_white[index_output, index_x, index_y] >= 0,
                    extracted_spectrum_white[index_output, index_x, index_y], 0)

                # Calculate the cost function according to equation B.8
                cost_function_white[index_output, index_x, index_y] = extracted_spectrum_white[
                                                                          index_output, index_x, index_y] * \
                                                                      matrix_c[
                                                                          index_output]

        # Reshape the cost function to the grid size
        # cost_function_white = cost_function_white.reshape((context.settings.grid_size, context.settings.grid_size) +
        #                                                   cost_function_white.shape[1:])
        # extracted_spectrum_white = extracted_spectrum_white.reshape(cost_function_white.shape)

        # Sum cost function over all wavelengths
        cost_function_white = np.sum(cost_function_white, axis=3)

        for index_output in range(len(matrix_b_elements)):
            j = cost_function_white[index_output, :, :]

            # Get extracted flux at positions around center at radius of maximum cost function
            height, width = cost_function_white[index_output, :, :].shape

            # Create a boolean mask for the circle
            center = (width // 2, height // 2)
            radius = np.sqrt((i1 - width // 2) ** 2 + (i2 - width // 2) ** 2)
            y, x = np.ogrid[:height, :width]
            mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2 + 0.5) & \
                   ((x - center[0]) ** 2 + (y - center[1]) ** 2 >= (radius - 1) ** 2 + 0.5)

            # Extract the pixels within the circle
            # circle_pixels = image[mask]
            # print(mask.shape)
            # print(extracted_photon_counts.shape)
            plt.imshow(cost_function_white[0, :, :] * mask)
            plt.colorbar()
            plt.show()
            a = np.einsum('ijk, ij -> ijk', extracted_spectrum_white[index_output, :, :], mask).reshape(
                context.settings.grid_size ** 2, -1)

            uncertainties = np.zeros(a.shape[1], dtype=object)

            for index in range(a.shape[1]):
                uncertainties[index] = np.std(a[:, index])

        context.extractions.append(Extraction(estimated_spectrum_cost_function_maximum, uncertainties, cost_function))

        return context
