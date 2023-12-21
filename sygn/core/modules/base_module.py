from abc import ABC, abstractmethod

from sygn.core.context import Context


class BaseModule(ABC):
    """Class representation of the base module.
    """
    dependencies: list

    @abstractmethod
    def apply(self, context: Context) -> Context:
        """Apply the module.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        pass
