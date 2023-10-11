from lifesim2.observatory.array_configurations import ArrayConfigurationEnum, \
    EmmaXCircularRotation, EmmaXDoubleStretch, EquilateralTriangleCircularRotation, RegularPentagonCircularRotation
from lifesim2.observatory.beam_combination_schemes import BeamCombinationSchemeEnum, \
    Kernel3, Kernel4, Kernel5, DoubleBracewell
from lifesim2.observatory.instrument_parameters import InstrumentParameters


class Observatory():
    """Class representation of an observatory.
    """

    def __init__(self):
        """Constructor method.
        """
        self._observatory_dict = None
        self.array_configuration = None
        self.beam_combination_scheme = None
        self.instrument_parameters = None

    def set_from_config(self, observatory_dictionary: dict()):
        """Read the configuration file and set the Observatory parameters.

        :param path_to_config_file: Path to the configuration file
        """
        self._observatory_dict = observatory_dictionary
        self._set_array_configuration_from_config()
        self._set_beam_combination_scheme_from_config()
        self._set_instrument_parameters_from_config()

    def _set_array_configuration_from_config(self):
        """Initialize the ArrayConfiguration object with the respective parameters.
        """
        baseline_minimum = self._observatory_dict['array_configuration']['baseline_minimum']
        baseline_maximum = self._observatory_dict['array_configuration']['baseline_maximum']
        baseline_ratio = self._observatory_dict['array_configuration']['baseline_ratio']
        modulation_period = self._observatory_dict['array_configuration']['modulation_period']

        array_configuration = self._observatory_dict['array_configuration']['type']

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
        beam_combination_scheme = self._observatory_dict['beam_combination_scheme']

        match beam_combination_scheme:
            case BeamCombinationSchemeEnum.DOUBLE_BRACEWELL.value:
                self.beam_combination_scheme = DoubleBracewell()

            case BeamCombinationSchemeEnum.KERNEL_3.value:
                self.beam_combination_scheme = Kernel3()

            case BeamCombinationSchemeEnum.KERNEL_4.value:
                self.beam_combination_scheme = Kernel4()

            case BeamCombinationSchemeEnum.KERNEL_5.value:
                self.beam_combination_scheme = Kernel5()

    def _set_instrument_parameters_from_config(self):
        """Initialize the InstrumentParameters object.
        """
        self.instrument_parameters = InstrumentParameters(
            parameters_dict=self._observatory_dict['instrument_parameters'])
