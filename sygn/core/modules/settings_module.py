from pathlib import Path

from sygn.core.contexts.base_context import BaseContext
from sygn.core.entities.settings import Settings
from sygn.core.modules.base_module import BaseModule
from sygn.io.config_reader import ConfigReader


class SettingsModule(BaseModule):
    """Class representation of the settings modules.
    """

    def __init__(self, path_to_config_file: Path):
        """Constructor method.

        :param path_to_config_file: Path to the config file
        """
        self.path_to_config_file = path_to_config_file
        self.settings = None

    def apply(self, context: BaseContext) -> BaseContext:
        """Apply the modules.

        :param context: The contexts object of the pipelines
        :return: The (updated) contexts object
        """
        config_dict = ConfigReader(path_to_config_file=self.path_to_config_file).get_dictionary_from_file()
        self.settings = Settings(**config_dict['settings'])
        context.settings = self.settings
        return context
