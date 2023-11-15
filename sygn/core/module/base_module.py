from abc import ABC, abstractmethod

from sygn.core.context.base_context import BaseContext


class BaseModule(ABC):
    """Class representation of the base module.
    """

    @abstractmethod
    def apply(self, context: BaseContext) -> BaseContext:
        """Apply the module.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        pass
