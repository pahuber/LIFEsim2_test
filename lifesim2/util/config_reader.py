import yaml
from astropy import units as u


class ConfigReader():
    """Class to read configuration files.
    """

    def __init__(self, path_to_config_file: str):
        """Constructor method.
        :param path_to_config_file: Path to the configuration file.
        """
        self.path_to_config_file = path_to_config_file
        self._config_dict = dict()
        self._read_raw_config_file()
        self._parse_units()

    def _read_raw_config_file(self):
        """Read the configuration file and save its content in a dictionary.
        """
        with open(self.path_to_config_file, 'r') as config_file:
            self._config_dict = yaml.load(config_file, Loader=yaml.SafeLoader)

    def _parse_units(self):
        """Parse the units of the numerical parameters and convert them to astropy quantities.
        """
        for key in self._config_dict['simulation'].keys():
            self._config_dict['simulation'][key] = u.Quantity(self._config_dict['simulation'][key])

        for key in self._config_dict['observation'].keys():
            self._config_dict['observation'][key] = u.Quantity(self._config_dict['observation'][key])

        for key in self._config_dict['observatory']['instrument_parameters']:
            self._config_dict['observatory']['instrument_parameters'][key] = \
                u.Quantity(self._config_dict['observatory']['instrument_parameters'][key])

    def get_config_from_file(self) -> dict():
        """Return a dictionary containing the content of the configuration file.
        :return: A dictionary with the configurations.
        """
        return self._config_dict
