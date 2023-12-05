from abc import ABC

import numpy as np

from sygn.core.context import Context


class MLMExtractionModule(ABC):
    """Class representation of the maximum likelihood method extraction module.
    """

    def apply(self, context: Context) -> Context:
        """Apply the module.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        # Create cost function and optimized flux array of shape (number of templates, number differential outputs,
        # number wavelengths)
        cost_function = np.zeros((len(context.templates), len(context.data), len(context.data[0])))
        optimized_flux = np.zeros((len(context.templates), len(context.data), len(context.data[0])))

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

                # Calculate optimized flux vector according to equation B.6
                optimized_flux[index_template][index_output] = np.diag(np.linalg.inv(matrix_b[index_output]) * matrix_c[
                    index_output])

                # Set positivity constraint
                optimized_flux[index_template][index_output] = np.where(
                    optimized_flux[index_template][index_output] >= 0, optimized_flux[index_template][index_output], 0)

                # Calculate the cost function according to equation B.8
                cost_function[index_template][index_output] = optimized_flux[index_template][index_output] * matrix_c[
                    index_output]

        # Calculate the total cost function over all wavelengths
        cost_function = np.sum(cost_function, axis=2)

        # Reshape the cost function to the grid size
        cost_function = cost_function.reshape(context.settings.grid_size, context.settings.grid_size,
                                              cost_function.shape[1])
        optimized_flux = optimized_flux.reshape(context.settings.grid_size, context.settings.grid_size,
                                                optimized_flux.shape[2])
        context.cost_function = cost_function
        context.optimized_flux = optimized_flux
        return context
