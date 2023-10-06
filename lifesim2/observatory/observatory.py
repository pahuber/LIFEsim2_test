from lifesim2.observatory.array_configurations import ArrayConfiguration, ArrayConfigurationEnum, \
    EmmaXCircularRotation, EmmaXDoubleStretch, EquilateralTriangleCircularRotation, RegularPentagonCircularRotation
from lifesim2.observatory.beam_combination_schemes import BeamCombinationScheme, BeamCombinationSchemeEnum, \
    DoubleBracewell4, Kernel3, Kernel4, Kernel5
from lifesim2.observatory.instrument_specification import InstrumentSpecification
from lifesim2.util.config_reader import ConfigReader


class Observatory():
    def __init__(self):
        self._config_dict = None
        self.array_configuration = None
        self.beam_combination_scheme = None
        self.instrument_specification = None

    def load_from_config(self, path_to_file: str):
        self._config_dict = ConfigReader(path_to_file).get_config_from_file()
        self._set_array_configuration_from_config()
        self._set_beam_combination_scheme_from_config()
        self._set_instrument_specification_from_config()

    def _set_array_configuration_from_config(self):
        array_configuration = self._config_dict['array_configuration']

        match array_configuration:
            case ArrayConfigurationEnum.EMMA_X_CIRCULAR_ROTATION.value:
                self.array_configuration = EmmaXCircularRotation()

            case ArrayConfigurationEnum.EMMA_X_DOUBLE_STRETCH.value:
                self.array_configuration = EmmaXDoubleStretch()

            case ArrayConfigurationEnum.EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION.value:
                self.array_configuration = EquilateralTriangleCircularRotation()

            case ArrayConfigurationEnum.REGULAR_PENTAGON_CIRCULAR_ROTATION.value:
                self.array_configuration = RegularPentagonCircularRotation()

    def _set_beam_combination_scheme_from_config(self):
        beam_combination_scheme = self._config_dict['beam_combination_scheme']

        match beam_combination_scheme:
            case BeamCombinationSchemeEnum.DOUBLE_BRACEWELL_4.value:
                self.beam_combination_scheme = DoubleBracewell4()

            case BeamCombinationSchemeEnum.KERNEL_3.value:
                self.beam_combination_scheme = Kernel3()

            case BeamCombinationSchemeEnum.KERNEL_4.value:
                self.beam_combination_scheme = Kernel4()

            case BeamCombinationSchemeEnum.KERNEL_5.value:
                self.beam_combination_scheme = Kernel5()

    def _set_instrument_specification_from_config(self):
        self.instrument_specification = InstrumentSpecification(specification_dict = self._config_dict['instrument_specification'])