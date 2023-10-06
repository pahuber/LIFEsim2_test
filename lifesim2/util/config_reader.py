import yaml
from astropy import units as u

class ConfigReader():
    def __init__(self, path_to_config_file: str):
        self.path_to_config_file = path_to_config_file
        self._config_dict = dict()
        self._read_raw_config_file()
        self._parse_instrument_specification_units()

    def _read_raw_config_file(self):
        with open(self.path_to_config_file, 'r') as config_file:
            self._config_dict = yaml.load(config_file, Loader=yaml.SafeLoader)

    def _parse_instrument_specification_units(self):
        for specification in self._config_dict['instrument_specification']:
            self._config_dict['instrument_specification'][specification] = \
                u.Quantity(self._config_dict['instrument_specification'][specification])

    def get_config_from_file(self):
        return self._config_dict
