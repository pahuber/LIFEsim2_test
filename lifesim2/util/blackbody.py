import astropy.units
import numpy as np
from astropy.modeling.models import BlackBody


def get_blackbody_spectrum(temperature: astropy.units.Quantity, spectral_range_lower_limit: astropy.units.Quantity,
                           spectral_range_upper_limit: astropy.units.Quantity) -> BlackBody:
    return BlackBody(temperature=temperature)(
        np.linspace(spectral_range_lower_limit.value, spectral_range_upper_limit.value,
                    1000) * spectral_range_upper_limit.unit)
