from pydantic import BaseModel


class NoiseContributions(BaseModel):
    """Class representation of the noise contributions that are considered for the simulation.
    """
    stellar_leakage: bool
    local_zodi_leakage: bool
    exozodi_leakage: bool
    fiber_injection_variability: bool
    optical_path_difference_variability: bool
