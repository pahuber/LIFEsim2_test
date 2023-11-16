from pathlib import Path

from sygn.core.contexts.generator_context import GeneratorContext
from sygn.core.entities.data_generator.data_generator import DataGenerator
from sygn.core.modules.mission_module import MissionModule
from sygn.core.modules.observatory_module import ObservatoryModule
from sygn.core.modules.planetary_system_module import PlanetarySystemModule
from sygn.core.modules.settings_module import SettingsModule
from sygn.core.pipelines.base_pipeline import BasePipeline
from sygn.io.fits_writer import FITSWriter
from sygn.util.grid import get_number_of_instances_in_list


class GeneratorPipeline(BasePipeline):
    """Class representation of the generator pipelines.
    """

    def __init__(self):
        """Constructor method.
        """
        super().__init__()
        self.context = GeneratorContext()
        self.data_generator = None

    def _generate_data(self):
        """Run the data generator to generate the synthetic data.
        """
        self.data_generator = DataGenerator(settings=self.context.settings,
                                            observation=self.context.observation,
                                            observatory=self.context.observatory,
                                            target_systems=self.context.photon_sources,
                                            time_range=self.context.time_range,
                                            animator=self.context.animator)
        self.data_generator.run()
        for index_target_system in range(len(self.data_generator.output)):
            self.context.data.append(
                self.data_generator.output[index_target_system].differential_photon_counts)

    def _validate_modules(self):
        """Validate that the correct number of each modules type is added to the pipelines.
        """
        for module_type in [SettingsModule, MissionModule, ObservatoryModule]:
            if not (get_number_of_instances_in_list(self._modules, module_type) == 1):
                raise TypeError(f'Need exactly one {module_type.__name__} per pipelines')
        if not (get_number_of_instances_in_list(self._modules, PlanetarySystemModule) >= 1):
            raise TypeError(f'Need at least one {module_type.__name__} per pipelines')

    def get_data(self) -> list:
        """Get the synthetic data.

        :return: The differential photon counts
        """
        return self.context.data

    def run(self):
        """Run the pipelines by calling the apply method of each modules.
        """
        self._validate_modules()
        for module in self._modules:
            context = module.apply(context=self.context)
        self._generate_data()

    def save_data_to_fits(self, output_path: Path, postfix: str = ''):
        """Save the differential photon counts to a FITS file.

        :param output_path: The output path of the FITS file
        :param postfix: Postfix to be appended to the output file name
        """
        FITSWriter.write_fits(output_path, postfix, self.context)
