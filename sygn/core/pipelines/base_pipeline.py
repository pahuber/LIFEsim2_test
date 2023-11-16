from abc import ABC, abstractmethod
from typing import Any

from sygn.core.contexts.base_context import BaseContext
from sygn.core.modules.base_module import BaseModule


class BasePipeline(ABC):
    """Class representation of the base pipelines.
    """

    def __init__(self):
        """Constructor method.
        """
        self._modules = []
        self.context = BaseContext()

    @abstractmethod
    def _validate_modules(self):
        """Validate that the correct number of each modules type is added to the pipelines.
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
        """Run the pipelines by calling the apply method of each modules.
        """
        pass

    def add_module(self, module: BaseModule):
        """Add a modules to the pipelines.

        :param module: The modules to be added
        """
        self._modules.append(module)
