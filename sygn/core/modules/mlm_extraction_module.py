import numpy as np
from astropy import units as u

from sygn.core.context import Context
from sygn.core.modules.base_module import BaseModule
from sygn.core.modules.data_generator_module import DataGeneratorModule
from sygn.core.modules.fits_reader_module import FITSReaderModule
from sygn.core.modules.template_generator_module import TemplateGeneratorModule
from sygn.util.helpers import FITSDataType


class MLExtractionModule(BaseModule):
    """Class representation of the maximum likelihood extraction module.
    """

    def __init__(self):
        """Constructor method.
        """
        self.dependencies = [(FITSReaderModule, FITSDataType.SyntheticMeasurement, FITSDataType.Template),
                             (FITSReaderModule, DataGeneratorModule, FITSDataType.Template),
                             (FITSReaderModule, TemplateGeneratorModule, FITSDataType.SyntheticMeasurement),
                             (DataGeneratorModule, TemplateGeneratorModule)]

    def apply(self, context: Context) -> Context:
        """Apply the module.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        # Create cost function and optimized flux array of shape (number of templates, number differential outputs,
        # number wavelengths)
        cost_function = np.zeros((len(context.templates), len(context.data), len(context.data[0])))
        estimated_flux = np.zeros((len(context.templates), len(context.data), len(context.data[0])))

        data_variance = np.var(context.data, axis=2)

        for index_template, template in enumerate(context.templates):
            # For each template and all outputs, calculate the matrix c and the elements of matrix B, according to
            # equations B.2 and B.3
            matrix_c = np.sum(context.data * template, axis=2) / data_variance
            matrix_b_elements = np.sum(template ** 2, axis=2) / data_variance
            matrix_b = np.zeros(matrix_b_elements.shape[0], dtype=object)

            for index_output in range(len(matrix_b_elements)):
                # For each output, create the diagonal matrix B
                matrix_b[index_output] = np.diag(matrix_b_elements[index_output])

                # Calculate estimated flux vector according to equation B.6
                estimated_flux[index_template][index_output] = np.diag(np.linalg.inv(matrix_b[index_output]) * matrix_c[
                    index_output])

                # Set positivity constraint
                estimated_flux[index_template][index_output] = np.where(
                    estimated_flux[index_template][index_output] >= 0, estimated_flux[index_template][index_output], 0)

                # Calculate the cost function according to equation B.8
                cost_function[index_template][index_output] = estimated_flux[index_template][index_output] * matrix_c[
                    index_output]

        # Reshape the cost function to the grid size
        cost_function = cost_function.reshape((context.settings.grid_size, context.settings.grid_size) +
                                              cost_function.shape[1:])
        estimated_flux = estimated_flux.reshape(cost_function.shape)

        # Sum cost function over all wavelengths
        cost_function = np.sum(cost_function, axis=3)

        # Get the estimated flux at the cost function maximum
        estimated_flux_cost_function_maximum = np.zeros(
            (context.observatory.beam_combination_scheme.number_of_differential_outputs,
             len(context.data[0]))) * u.ph

        for index_output in range(len(estimated_flux_cost_function_maximum)):
            j = cost_function[:, :, index_output]
            j[np.isnan(j)] = 0
            j_max = np.max(j)

            index = np.where(j == j_max)
            i1, i2 = index[0][0], index[1][0]
            estimated_flux_cost_function_maximum[index_output] = estimated_flux[i1, i2, index_output] * u.ph

        context.cost_function = cost_function
        context.optimized_flux = estimated_flux_cost_function_maximum
        return context
