from lifesim2.observatory.array_configurations import ArrayConfigurationEnum, \
    EmmaXCircularRotation, EmmaXDoubleStretch, EquilateralTriangleCircularRotation, RegularPentagonCircularRotation
from lifesim2.observatory.beam_combination_schemes import BeamCombinationSchemeEnum, \
    DoubleBracewell4, Kernel3, Kernel4, Kernel5
from lifesim2.observatory.instrument_specification import InstrumentSpecification
from lifesim2.util.config_reader import ConfigReader


class Observatory():
    """Class representation of an observatory.
    """

    def __init__(self):
        """Constructor method.
        """
        self._config_dict = None
        self.array_configuration = None
        self.beam_combination_scheme = None
        self.instrument_specification = None

    def load_from_config(self, path_to_file: str):
        """Read the configuration file and set the Observatory parameters.
        :param path_to_file: Path to the configuration file
        """
        self._config_dict = ConfigReader(path_to_file).get_config_from_file()
        self._set_array_configuration_from_config()
        self._set_beam_combination_scheme_from_config()
        self._set_instrument_specification_from_config()

    def _set_array_configuration_from_config(self):
        """Initialize the ArrayConfiguration object with the respective parameters.
        """
        baseline_minimum = self._config_dict['instrument_specification']['baseline_minimum']
        baseline_maximum = self._config_dict['instrument_specification']['baseline_maximum']
        baseline_ratio = self._config_dict['instrument_specification']['baseline_ratio']
        modulation_period = self._config_dict['instrument_specification']['modulation_period']

        array_configuration = self._config_dict['array_configuration']

        match array_configuration:
            case ArrayConfigurationEnum.EMMA_X_CIRCULAR_ROTATION.value:
                self.array_configuration = EmmaXCircularRotation(baseline_minimum=baseline_minimum,
                                                                 baseline_maximum=baseline_maximum,
                                                                 baseline_ratio=baseline_ratio,
                                                                 modulation_period=modulation_period)

            case ArrayConfigurationEnum.EMMA_X_DOUBLE_STRETCH.value:
                self.array_configuration = EmmaXDoubleStretch(baseline_minimum=baseline_minimum,
                                                              baseline_maximum=baseline_maximum,
                                                              baseline_ratio=baseline_ratio,
                                                              modulation_period=modulation_period)

            case ArrayConfigurationEnum.EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION.value:
                self.array_configuration = EquilateralTriangleCircularRotation(baseline_minimum=baseline_minimum,
                                                                               baseline_maximum=baseline_maximum,
                                                                               baseline_ratio=baseline_ratio,
                                                                               modulation_period=modulation_period)

            case ArrayConfigurationEnum.REGULAR_PENTAGON_CIRCULAR_ROTATION.value:
                self.array_configuration = RegularPentagonCircularRotation(baseline_minimum=baseline_minimum,
                                                                           baseline_maximum=baseline_maximum,
                                                                           baseline_ratio=baseline_ratio,
                                                                           modulation_period=modulation_period)

    def _set_beam_combination_scheme_from_config(self):
        """Initialize the BeamCombinationScheme object.
        """
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
        """Initialize the InstrumentSpecification object.
        """
        self.instrument_specification = InstrumentSpecification(
            specification_dict=self._config_dict['instrument_specification'])
