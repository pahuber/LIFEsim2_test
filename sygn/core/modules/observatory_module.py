from pathlib import Path

from sygn.core.contexts.base_context import BaseContext
from sygn.core.entities.observatory.array_configurations import ArrayConfiguration, ArrayConfigurationEnum, \
    EmmaXCircularRotation, EmmaXDoubleStretch, EquilateralTriangleCircularRotation, RegularPentagonCircularRotation
from sygn.core.entities.observatory.beam_combination_schemes import BeamCombinationScheme, BeamCombinationSchemeEnum, \
    DoubleBracewell, Kernel3, Kernel4, Kernel5
from sygn.core.entities.observatory.instrument_parameters import InstrumentParameters
from sygn.core.entities.observatory.observatory import Observatory
from sygn.core.modules.base_module import BaseModule
from sygn.io.config_reader import ConfigReader


class ObservatoryModule(BaseModule):
    """Class representation of the obsrvatory modules.
    """

    def __init__(self, path_to_config_file: Path):
        """Constructor method.

        :param path_to_config_file: Pth to the config file
        """
        self.path_to_config_file = path_to_config_file
        self.observatory = None

    def _initialize_array_configuration_from_config(self, config_dict: dict) -> ArrayConfiguration:
        """Return an ArrayConfiguration object.

        :param config_dict: The config dictionary
        :return: ArrayConfiguration object.
        """
        type = config_dict['observatory']['array_configuration']['type']

        match type:
            case ArrayConfigurationEnum.EMMA_X_CIRCULAR_ROTATION.value:
                return EmmaXCircularRotation(**config_dict['observatory']['array_configuration'])

            case ArrayConfigurationEnum.EMMA_X_DOUBLE_STRETCH.value:
                return EmmaXDoubleStretch(**config_dict['observatory']['array_configuration'])

            case ArrayConfigurationEnum.EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION.value:
                return EquilateralTriangleCircularRotation(**config_dict['observatory']['array_configuration'])

            case ArrayConfigurationEnum.REGULAR_PENTAGON_CIRCULAR_ROTATION.value:
                return RegularPentagonCircularRotation(**config_dict['observatory']['array_configuration'])

    def _initialize_beam_combination_scheme_from_config(self, config_dict: dict) -> BeamCombinationScheme:
        """Return an BeamCombinationScheme object.

        :param config_dict: The config dictionary
        :return: BeamCombinationScheme object.
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

    def apply(self, context: BaseContext) -> BaseContext:
        """Apply the modules.

        :param context: The contexts object of the pipelines
        :return: The (updated) contexts object
        """
        config_dict = ConfigReader(path_to_config_file=self.path_to_config_file).get_dictionary_from_file()
        self.observatory = Observatory()
        self.observatory.array_configuration = self._initialize_array_configuration_from_config(config_dict)
        self.observatory.beam_combination_scheme = self._initialize_beam_combination_scheme_from_config(config_dict)
        self.observatory.instrument_parameters = InstrumentParameters(
            **config_dict['observatory']['instrument_parameters'])
        context.observatory = self.observatory
        return context
