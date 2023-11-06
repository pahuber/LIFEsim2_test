from abc import ABC, abstractmethod

from sygn.core.context import Context


class BaseModule(ABC):

    @abstractmethod
    def apply(self, context: Context) -> Context:
        pass
