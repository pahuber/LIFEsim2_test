import numpy as np

from sygn.core.context import Context
from sygn.core.entities.photon_sources.planet import Planet
from sygn.core.extraction import Extraction
from sygn.core.modules.animator_module import AnimatorModule
from sygn.core.modules.base_module import BaseModule
from sygn.core.modules.config_loader_module import ConfigLoaderModule
from sygn.core.modules.data_generator_module import DataGeneratorModule
from sygn.core.modules.fits_reader_module import FITSReaderModule
from sygn.core.modules.mlm_extraction_module import MLExtractionModule
from sygn.core.modules.target_loader_module import TargetLoaderModule
from sygn.core.modules.template_generator_module import TemplateGeneratorModule
from sygn.util.grid import get_number_of_instances_in_list
from sygn.util.helpers import FITSReadWriteType


class Pipeline():
    """Class representation of the pipeline.
    """

    def __init__(self):
        """Constructor method.
        """
        self._modules = []
        self._context = Context()

    def _check_data_template_generation_reading(self):
        """Check that there is no combination of DataGenerationModule and TemplateGeneratorModule or FITSReaderModule.
        """
        if (get_number_of_instances_in_list(self._modules, FITSReaderModule) > 0
                and get_number_of_instances_in_list(self._modules, DataGeneratorModule) == 1):
            for module in self._modules:
                if isinstance(module, FITSReaderModule) and module._data_type == FITSReadWriteType.SyntheticMeasurement:
                    raise TypeError(
                        f'Can not use DataGenerationModule and FITSReaderModule with FITSDataType.SyntheticMeasurements at the same time')
        if (get_number_of_instances_in_list(self._modules, FITSReaderModule) > 0
                and get_number_of_instances_in_list(self._modules, TemplateGeneratorModule) == 1):
            for module in self._modules:
                if isinstance(module, FITSReaderModule) and module._data_type == FITSReadWriteType.Template:
                    raise TypeError(
                        f'Can not use TemplateGeneratorModule and FITSReaderModule with FITSDataType.Template at the same time')

    def _check_module_dependencies(self):
        """Check that all modules have their dependencies satisfied, i.e. that for each module all required modules are
        present in the pipeline and if there is a specific FITSDataType required, that there is a module using that data
        type."""
        for module in self._modules:
            options_fulfilled = 0
            # Check all possible options of allowed dependency combinations
            for option in module.dependencies:
                dependencies_fulfilled = 0
                # Check if all dependencies of an option are fulfilled
                for dependency in option:
                    # Check that if dependency is a FITSDataType, there is a FITSReaderModule or FITSWriterModule with
                    # that data type
                    if isinstance(dependency, FITSReadWriteType):
                        for module_2 in self._modules:
                            if isinstance(module_2, FITSReaderModule) and module_2._data_type == dependency:
                                dependencies_fulfilled += 1
                    # Check that if dependency is a BaseModule, there is a module of that type
                    elif issubclass(dependency, BaseModule):
                        for module_2 in self._modules:
                            if isinstance(module_2, dependency):
                                dependencies_fulfilled += 1
                                break
                # If all dependencies of an option are fulfilled, indicate that one option os fulfilled
                if dependencies_fulfilled == len(option):
                    options_fulfilled += 1
            # If not exactly one option is fulfilled, raise an error
            if options_fulfilled != 1 and len(module.dependencies) > 0:
                raise TypeError(f'{module.__class__.__name__} has unsatisfied module dependencies')

    def _check_number_of_modules(self):
        """Check that there is at most one module of each type in the pipeline.
        """
        for module_type in [ConfigLoaderModule, TargetLoaderModule, AnimatorModule, DataGeneratorModule,
                            TemplateGeneratorModule, MLExtractionModule]:
            if not (get_number_of_instances_in_list(self._modules, module_type) <= 1):
                raise TypeError(f'Can not have more than one {module_type.__name__} per pipeline')

    def _validate_modules(self):
        """Validate that th modules of the pipeline are correct.
        """
        self._check_number_of_modules()
        self._check_data_template_generation_reading()
        self._check_module_dependencies()

    def get_signal(self) -> np.ndarray:
        """Return the signal.

        :return: An array containing the signal
        """
        return self._context.signal

    def get_extractions(self) -> list[Extraction]:
        """Return the extractions.

        :return: A list containing the extractions
        """
        return self._context.extractions

    def get_planets(self) -> list[Planet]:
        """Return the planets.

        :return: A list containing the planets
        """
        return [source for source in self._context.photon_sources if isinstance(source, Planet)]

    def get_wavelengths(self) -> np.ndarray:
        """Return the wavelengths.

        :return: An array containing the wavelengths
        """
        return self._context.observatory.instrument_parameters.wavelength_bin_centers

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
