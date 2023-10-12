from abc import ABC


class Source(ABC):
    def __init__(self):
        super.__init__()
        self.flux = None

    # @abstractmethod
    # def get_spectrum(self):
    #     pass
