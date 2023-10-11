from astropy import units as u


class InstrumentParameters():
    def __init__(self, parameters_dict: dict() = None):
        self.aperture_radius = 4 * u.m / 2
        self._config_dict = None

    # def set_from_config(self):
