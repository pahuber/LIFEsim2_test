import numpy as np

from sygn.core.context import Context
from sygn.core.modules.base_module import BaseModule
from sygn.core.modules.config_loader_module import ConfigLoaderModule
from sygn.util.grid import get_number_of_instances_in_list


class Pipeline():
    """Class representation of the pipeline.
    """

    def __init__(self):
        """Constructor method.
        """
        self._modules = []
        self._context = Context()

    def _validate_modules(self):
        """Validate that the correct number of each modules type is added to the pipelines.
        """
        for module_type in [ConfigLoaderModule]:
            if not (get_number_of_instances_in_list(self._modules, module_type) == 1):
                raise TypeError(f'Need exactly one {module_type.__name__} per pipeline')

    def get_data(self) -> np.ndarray:
        """Return the data.

        :return: An array containing the data
        """
        return self._context.data

    def run(self):
        """Run the pipeline by calling the apply method of each module.
        """
        self._validate_modules()
        for module in self._modules:
            self._context = module.apply(context=self._context)

    def add_module(self, module: BaseModule):
        """Add a modules to the pipelines.

        :param module: The modules to be added
        """
        self._modules.append(module)
