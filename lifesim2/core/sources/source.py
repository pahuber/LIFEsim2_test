from abc import ABC

import astropy

from lifesim2.util.blackbody import get_blackbody_spectrum


class Source(ABC):
    def __init__(self):
        super.__init__()
        self.name = None
        self.flux = None
        self.temperature = None

    def create_blackbody_spectrum(self,
                                  spectral_range_lower_limit: astropy.units.Quantity,
                                  spectral_range_upper_limit: astropy.units.Quantity):
        self.spectrum = get_blackbody_spectrum(temperature=self.temperature,
                                               spectral_range_lower_limit=spectral_range_lower_limit,
                                               spectral_range_upper_limit=spectral_range_upper_limit)

    # @abstractmethod
    # def get_spectrum(self):
    #     pass
