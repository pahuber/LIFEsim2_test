import astropy.units


class InstrumentParameters():
    def __init__(self,
                 aperture_diameter: astropy.units.Quantity,
                 spectral_range_lower_limit: astropy.units.Quantity,
                 spectral_range_upper_limit: astropy.units.Quantity,
                 spectral_resolution: astropy.units.Quantity):
        self.aperture_radius = aperture_diameter / 2
        self.spectral_range_lower_limit = spectral_range_lower_limit
        self.spectral_range_upper_limit = spectral_range_upper_limit
        self.spectral_resolution = spectral_resolution
