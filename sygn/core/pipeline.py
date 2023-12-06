import numpy as np

from sygn.core.context import Context
from sygn.core.modules.animator_module import AnimatorModule
from sygn.core.modules.base_module import BaseModule
from sygn.core.modules.config_loader_module import ConfigLoaderModule
from sygn.core.modules.data_generator_module import DataGeneratorModule
from sygn.core.modules.fits_reader_module import FITSReaderModule
from sygn.core.modules.fits_writer_module import FITSWriterModule
from sygn.core.modules.mlm_extraction_module import MLMExtractionModule
from sygn.core.modules.target_loader_module import TargetLoaderModule
from sygn.core.modules.template_generator_module import TemplateGeneratorModule
from sygn.util.grid import get_number_of_instances_in_list
from sygn.util.helpers import FITSDataType


class Pipeline():
    """Class representation of the pipeline.
    """

    def __init__(self):
        """Constructor method.
        """
        self._modules = []
        self._context = Context()

    def _validate_modules(self):
        """Validate that th modules of the pipeline are correct.
        """
        # Check that number of modules is correct
        for module_type in [ConfigLoaderModule, TargetLoaderModule, AnimatorModule, DataGeneratorModule,
                            TemplateGeneratorModule, MLMExtractionModule]:
            if not (get_number_of_instances_in_list(self._modules, module_type) <= 1):
                raise TypeError(f'Can not have more than one {module_type.__name__} per pipeline')

        # Check that module dependencies are satisfied
        if (get_number_of_instances_in_list(self._modules, DataGeneratorModule) == 1
                and not (get_number_of_instances_in_list(self._modules, ConfigLoaderModule) == 1
                         and get_number_of_instances_in_list(self._modules, TargetLoaderModule) == 1)):
            raise TypeError(
                f'Can not use DataGeneratorModule without ConfigLoaderModule and TargetLoaderModule')
        if (get_number_of_instances_in_list(self._modules, ConfigLoaderModule) == 1
                and not (get_number_of_instances_in_list(self._modules, DataGeneratorModule) == 1
                         and get_number_of_instances_in_list(self._modules, TargetLoaderModule) == 1)):
            raise TypeError(
                f'Can not use ConfigLoaderModule without DataGeneratorModule and TargetLoaderModule')
        if (get_number_of_instances_in_list(self._modules, TargetLoaderModule) == 1
                and not (get_number_of_instances_in_list(self._modules, DataGeneratorModule) == 1
                         and get_number_of_instances_in_list(self._modules, ConfigLoaderModule) == 1)):
            raise TypeError(
                f'Can not use TargetLoaderModule without DataGeneratorModule and ConfigLoaderModule')
        if (get_number_of_instances_in_list(self._modules, AnimatorModule) == 1
                and not (get_number_of_instances_in_list(self._modules, DataGeneratorModule) == 1)):
            raise TypeError(f'Can not use AnimatorModule without DataGeneratorModule')
        if (get_number_of_instances_in_list(self._modules, FITSWriterModule) == 1
                and not (get_number_of_instances_in_list(self._modules, DataGeneratorModule) == 1
                         or get_number_of_instances_in_list(self._modules, TemplateGeneratorModule) == 1)):
            raise TypeError(f'Can not use FITSWriterModule without DataGeneratorModule or TemplateGeneratorModule')

        # Check that data/templates are not generated and read at the same time
        if (get_number_of_instances_in_list(self._modules, FITSReaderModule) > 0
                and get_number_of_instances_in_list(self._modules, DataGeneratorModule) == 1):
            for module in get_number_of_instances_in_list(self._modules, FITSReaderModule):
                if module.data_type == FITSDataType.SyntheticMeasurement:
                    raise TypeError(
                        f'Can not use DataGenerationModule and FITSReaderModule with FITSDataType.SyntheticMeasurements at the same time')
        if (get_number_of_instances_in_list(self._modules, FITSReaderModule) > 0
                and get_number_of_instances_in_list(self._modules, TemplateGeneratorModule) == 1):
            for module in get_number_of_instances_in_list(self._modules, FITSReaderModule):
                if module.data_type == FITSDataType.Template:
                    raise TypeError(
                        f'Can not use TemplateGeneratorModule and FITSReaderModule with FITSDataType.Template at the same time')

    def get_data(self) -> np.ndarray:
        """Return the data.

        :return: An array containing the data
        """
        return self._context.data

    def run(self):
        """Run the pipeline by calling the apply method of each module.
        """
        self._validate_modules()
        for module in self._modules:
            self._context = module.apply(context=self._context)

    def add_module(self, module: BaseModule):
        """Add a modules to the pipelines.

        :param module: The modules to be added
        """
        self._modules.append(module)
