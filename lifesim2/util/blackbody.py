import astropy.units
import numpy as np
from astropy.modeling.models import BlackBody


def get_blackbody_spectrum(temperature: astropy.units.Quantity, wavelength_range: np.ndarray) -> BlackBody:
    return BlackBody(temperature=temperature)(wavelength_range)
