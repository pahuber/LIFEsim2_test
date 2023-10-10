from astropy import units as u


class InstrumentParameters():
    def __init__(self, specification_dict: dict() = None):
        self.aperture_radius = 4 * u.m / 2
