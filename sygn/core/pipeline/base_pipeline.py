from abc import ABC, abstractmethod
from typing import Any

from sygn.core.context.base_context import BaseContext
from sygn.core.module.base_module import BaseModule


class BasePipeline(ABC):
    """Class representation of the base pipeline.
    """

    def __init__(self):
        """Constructor method.
        """
        self.modules = []
        self.context = BaseContext()

    @abstractmethod
    def _validate_modules(self):
        """Validate that the correct number of each module type is added to the pipeline.
        """
        pass

    @abstractmethod
    def get_data(self) -> Any:
        """Get the synthetic data.

        :return: The data
        """
        pass

    @abstractmethod
    def run(self):
        """Run the pipeline by calling the apply method of each module.
        """
        pass

    def add_module(self, module: BaseModule):
        """Add a module to the pipeline.

        :param module: The module to be added
        """
        self.modules.append(module)
