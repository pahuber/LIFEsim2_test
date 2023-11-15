from pathlib import Path

from sygn.core.pipeline.base_pipeline import BasePipeline
from sygn.core.context.generator_context import GeneratorContext
from sygn.core.module.data_generator.data_generator_module import DataGeneratorModule
from sygn.core.module.observation.observation_module import ObservationModule
from sygn.core.module.observatory.observatory_module import ObservatoryModule
from sygn.core.module.settings.settings_module import SettingsModule
from sygn.core.module.target_system.target_system_module import TargetSystemModule
from sygn.io.fits_writer import FITSWriter
from sygn.util.grid import get_number_of_instances_in_list


class GeneratorPipeline(BasePipeline):
    """Class representation of the generator pipeline.
    """

    def __init__(self):
        """Constructor method.
        """
        super().__init__()
        self.context = GeneratorContext()

    def _validate_modules(self):
        """Validate that the correct number of each module type is added to the pipeline.
        """
        for module_type in [SettingsModule, ObservationModule, ObservatoryModule, DataGeneratorModule]:
            if not (get_number_of_instances_in_list(self.modules, module_type) == 1):
                raise TypeError(f'Need exactly one {module_type.__name__} per pipeline')
        if not (get_number_of_instances_in_list(self.modules, TargetSystemModule) >= 1):
            raise TypeError(f'Need at least one {module_type.__name__} per pipeline')

    def get_data(self) -> list:
        """Get the synthetic data.

        :return: The differential photon counts
        """
        return self.context.differential_photon_counts_list

    def save_data_to_fits(self, output_path: Path, postfix: str = ''):
        """Save the differential photon counts to a FITS file.

        :param output_path: The output path of the FITS file
        :param postfix: Postfix to be appended to the output file name
        """
        FITSWriter.write_fits(output_path, postfix, self.context)
