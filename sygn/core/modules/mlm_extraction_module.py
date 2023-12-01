from abc import ABC, abstractmethod

from sygn.core.context import Context


class MLMExtractionModule(ABC):
    """Class representation of the maximum likelihood method extraction module.
    """

    @abstractmethod
    def apply(self, context: Context) -> Context:
        """Apply the module.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        pass
