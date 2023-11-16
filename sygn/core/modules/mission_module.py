from pathlib import Path

from sygn.core.contexts.base_context import BaseContext
from sygn.core.entities.mission import Mission
from sygn.core.modules.base_module import BaseModule
from sygn.io.config_reader import ConfigReader


class MissionModule(BaseModule):
    """Class representation of the mission modules.
    """

    def __init__(self, path_to_config_file: Path):
        """Constructor method.

        :param path_to_config_file: Path to the config file
        """
        self.path_to_config_file = path_to_config_file
        self.observation = None

    def apply(self, context: BaseContext) -> BaseContext:
        """Apply the modules.

        :param context: The contexts object of the pipelines
        :return: The (updated) contexts object
        """
        config_dict = ConfigReader(path_to_config_file=self.path_to_config_file).get_dictionary_from_file()
        self.observation = Mission(**config_dict['mission'])
        context.observation = self.observation
        return context
