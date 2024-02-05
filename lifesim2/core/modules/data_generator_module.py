from datetime import datetime

import numpy as np
from astropy import units as u

from lifesim2.core.context import Context
from lifesim2.core.entities.photon_sources.planet import Planet
from lifesim2.core.modules.base_module import BaseModule
from lifesim2.core.modules.config_loader_module import ConfigLoaderModule
from lifesim2.core.modules.target_loader_module import TargetLoaderModule
from lifesim2.core.processing.data_generation import DataGenerator, GenerationMode


def get_dictionary_from_list_containing_key(source_name, target_systems):
    pass


class DataGeneratorModule(BaseModule):
    """Class representation of the data generator modules.
    """

    def __init__(self):
        """Constructor method.

        :param path_to_config_file: Path to the config file
        """
        self.dependencies = [(ConfigLoaderModule, TargetLoaderModule)]

    def _create_animation(self, context: Context):
        """Prepare the animation writer and generate the data.
        """
        planet = [source for source in context.photon_sources if
                  (isinstance(source, Planet) and source.name == context.animator.planet_name)][0]

        context.animator.prepare_animation_writer(context.time_range, context.settings.grid_size, planet)

        with context.animator.writer.saving(context.animator.figure,
                                            f"animation_{context.animator.planet_name}_{np.round(context.animator.closest_wavelength.to(u.um).value, 3)}um_{datetime.now().strftime('%Y%m%d_%H%M%S.%f')}.gif",
                                            300):
            data_generator = DataGenerator(context, GenerationMode.data)
            return data_generator.generate_data()

    def apply(self, context: Context) -> Context:
        """Apply the modules.

        :param context: The contexts object of the pipelines
        :return: The (updated) contexts object
        """
        context.observatory.set_optimal_baseline(context.star,
                                                 context.mission.optimized_differential_output,
                                                 context.mission.optimized_wavelength,
                                                 context.mission.optimized_star_separation,
                                                 context.mission.baseline_minimum,
                                                 context.mission.baseline_maximum)
        if context.animator:
            context.signal, _ = self._create_animation(context)
        else:
            data_generator = DataGenerator(context, GenerationMode.data)
            context.signal, _ = data_generator.generate_data()
        return context
