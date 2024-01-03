from pathlib import Path

from sygn.core.context import Context
from sygn.core.entities.mission import Mission
from sygn.core.entities.observatory.array_configurations import ArrayConfiguration, ArrayConfigurationEnum, \
    EmmaXCircularRotation, EmmaXDoubleStretch, EquilateralTriangleCircularRotation, RegularPentagonCircularRotation
from sygn.core.entities.observatory.beam_combination_schemes import BeamCombinationScheme, DoubleBracewell, Kernel3, \
    Kernel4, Kernel5, BeamCombinationSchemeEnum
from sygn.core.entities.observatory.instrument_parameters import InstrumentParameters
from sygn.core.entities.observatory.observatory import Observatory
from sygn.core.entities.settings import Settings
from sygn.core.modules.base_module import BaseModule
from sygn.io.config_reader import ConfigReader


class ConfigLoaderModule(BaseModule):
    """Class representation of the config loader module.
    """

    def __init__(self,
                 path_to_config_file: Path,
                 settings: Settings = None,
                 mission: Mission = None,
                 observatory: Observatory = None,
                 config_dict: dict = None):
        """Constructor method.

        :param path_to_config_file: Path to the configuration file
        :param settings: The settings object
        :param mission: The mission object
        :param observatory: The observatory object
        """
        self._path_to_config_file = path_to_config_file
        self._config_dict = config_dict
        self.settings = settings
        self.mission = mission
        self.observatory = observatory
        self.dependencies = []

    def _load_array_configuration(self, config_dict: dict) -> ArrayConfiguration:
        """Return the array configuration object from the dictionary.

        :param config_dict: The dictionary
        :return: The array configuration object.
        """
        type = config_dict['observatory']['array_configuration']

        match type:
            case ArrayConfigurationEnum.EMMA_X_CIRCULAR_ROTATION.value:
                return EmmaXCircularRotation(modulation_period=self.mission.modulation_period,
                                             baseline_ratio=self.mission.baseline_ratio)

            case ArrayConfigurationEnum.EMMA_X_DOUBLE_STRETCH.value:
                return EmmaXDoubleStretch(modulation_period=self.mission.modulation_period,
                                          baseline_ratio=self.mission.baseline_ratio)

            case ArrayConfigurationEnum.EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION.value:
                return EquilateralTriangleCircularRotation(modulation_period=self.mission.modulation_period,
                                                           baseline_ratio=self.mission.baseline_ratio)

            case ArrayConfigurationEnum.REGULAR_PENTAGON_CIRCULAR_ROTATION.value:
                return RegularPentagonCircularRotation(modulation_period=self.mission.modulation_period,
                                                       baseline_ratio=self.mission.baseline_ratio)

    def _load_beam_combination_scheme(self, config_dict: dict) -> BeamCombinationScheme:
        """Return the beam combination scheme object from the dictionary.

        :param config_dict: The dictionary
        :return: The beam combination object.
        """
        beam_combination_scheme = config_dict['observatory']['beam_combination_scheme']

        match beam_combination_scheme:
            case BeamCombinationSchemeEnum.DOUBLE_BRACEWELL.value:
                return DoubleBracewell()

            case BeamCombinationSchemeEnum.KERNEL_3.value:
                return Kernel3()

            case BeamCombinationSchemeEnum.KERNEL_4.value:
                return Kernel4()

            case BeamCombinationSchemeEnum.KERNEL_5.value:
                return Kernel5()

    def _load_mission(self, config_dict: dict) -> Mission:
        """Return the mission object from the dictionary.

        :param config_dict: The dictionary
        :return: The mission object
        """
        return Mission(**config_dict['mission'])

    def _load_observatory(self, config_dict: dict) -> Observatory:
        """Return the observatory object from the dictionary.

        :param config_dict: The dictionary
        :return: The observatory object
        """
        observatory = Observatory()
        observatory.array_configuration = self._load_array_configuration(config_dict)
        observatory.beam_combination_scheme = self._load_beam_combination_scheme(config_dict)
        observatory.instrument_parameters = InstrumentParameters(**config_dict['observatory']['instrument_parameters'])
        return observatory

    def _load_settings(self, config_dict: dict) -> Settings:
        """Return the settings object from the dictionary.

        :param config_dict: The dictionary
        :return: The settings object
        """
        return Settings(**config_dict['settings'], integration_time=self.mission.integration_time)

    def apply(self, context: Context) -> Context:
        """Load the configurations from the config file and initialize the settings, mission and observatory objects.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        if not self._config_dict:
            self._config_dict = ConfigReader(path_to_config_file=self._path_to_config_file).get_dictionary_from_file()
        self.mission = self._load_mission(self._config_dict) if self.mission is None else self.mission
        self.settings = self._load_settings(self._config_dict) if self.settings is None else self.settings
        self.observatory = self._load_observatory(self._config_dict) if self.observatory is None else self.observatory
        context.settings = self.settings
        context.mission = self.mission
        context.observatory = self.observatory
        return context
