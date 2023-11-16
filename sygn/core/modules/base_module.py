from abc import ABC, abstractmethod

from sygn.core.contexts.base_context import BaseContext


class BaseModule(ABC):
    """Class representation of the base modules.
    """

    @abstractmethod
    def apply(self, context: BaseContext) -> BaseContext:
        """Apply the modules.

        :param context: The contexts object of the pipelines
        :return: The (updated) contexts object
        """
        pass
