from pathlib import Path

from sygn.core.context import Context
from sygn.core.modules.base_module import BaseModule
from sygn.core.modules.settings.settings import Settings
from sygn.io.config_reader import ConfigReader


class SettingsModule(BaseModule):
    """Class representation of the settings module.
    """

    def __init__(self, path_to_config_file: Path):
        """Constructor method.

        :param path_to_config_file: Path to the config file
        """
        self.path_to_config_file = path_to_config_file
        self.settings = None

    def apply(self, context: Context) -> Context:
        """Apply the module.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        config_dict = ConfigReader(path_to_config_file=self.path_to_config_file).get_dictionary_from_file()
        self.settings = Settings(**config_dict['settings'])
        context.settings = self.settings
        return context
