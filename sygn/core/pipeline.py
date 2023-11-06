from pathlib import Path

from sygn.core.context import Context
from sygn.core.modules.base_module import BaseModule
from sygn.core.modules.data_generator.data_generator_module import DataGeneratorModule
from sygn.core.modules.observation.observation_module import ObservationModule
from sygn.core.modules.observatory.observatory_module import ObservatoryModule
from sygn.core.modules.settings.settings_module import SettingsModule
from sygn.core.modules.target_system.target_system_module import TargetSystemModule
from sygn.io.fits_writer import FITSWriter
from sygn.util.grid import get_number_of_instances_in_list


class Pipeline():

    def __init__(self):
        self.modules = []
        self.context = Context()

    def _validate_modules(self):
        for module_type in [SettingsModule, ObservationModule, ObservatoryModule, DataGeneratorModule]:
            if not (get_number_of_instances_in_list(self.modules, module_type) == 1):
                raise TypeError(f'Need exactly one {module_type.__name__} per pipeline')
        if not (get_number_of_instances_in_list(self.modules, TargetSystemModule) >= 1):
            raise TypeError(f'Need at least one {module_type.__name__} per pipeline')

    def add_module(self, module: BaseModule):
        self.modules.append(module)

    def get_data(self):
        return self.context.differential_photon_counts

    def run(self):
        self._validate_modules()
        for module in self.modules:
            context = module.apply(context=self.context)

    def save_data_to_fits(self, output_path: Path, postfix: str = ''):
        """Save the differential photon counts to a FITS file.

        :param output_path: The output path of the FITS file
        :param postfix: Postfix to be appended to the output file name
        """
        FITSWriter.write_fits(output_path, postfix, self.context)
