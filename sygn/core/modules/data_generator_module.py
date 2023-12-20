from datetime import datetime

import numpy as np
from astropy import units as u

from sygn.core.context import Context
from sygn.core.entities.photon_sources.planet import Planet
from sygn.core.modules.base_module import BaseModule
from sygn.core.modules.config_loader_module import ConfigLoaderModule
from sygn.core.modules.target_loader_module import TargetLoaderModule
from sygn.core.processing.data_generation import DataGenerator, GenerationMode


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

    def _update_animation_frame(self, time, intensity_responses, pair_of_indices, index_pair, index_photon_source,
                                source, wavelength, index_time):
        self.animator.update_collector_position(time, self.observatory)
        self.animator.update_differential_intensity_response(
            intensity_responses[pair_of_indices[0]] - intensity_responses[pair_of_indices[1]])
        self.animator.update_differential_photon_counts(
            self.output[index_photon_source].differential_photon_counts_by_source[index_pair][
                source.name][
                wavelength][
                index_time], index_time)
        self.animator.writer.grab_frame()

    def apply(self, context: Context) -> Context:
        """Apply the modules.

        :param context: The contexts object of the pipelines
        :return: The (updated) contexts object
        """
        context.observatory.array_configuration.set_optimal_baseline(context.mission.optimized_wavelength,
                                                                     context.star.habitable_zone_central_angular_radius)
        if context.animator:
            context.signal, _ = self._create_animation(context)
        else:
            data_generator = DataGenerator(context, GenerationMode.data)
            context.signal, _ = data_generator.generate_data()
        return context

    def _create_animation(self, context: Context):
        """Prepare the animation writer and run the time loop.
        """
        planet = [source for source in context.photon_sources if
                  (isinstance(source, Planet) and source.name == context.animator.planet_name)][0]
        context.animator.prepare_animation_writer(context.time_range,
                                                  context.settings.grid_size,
                                                  planet)
        with context.animator.writer.saving(context.animator.figure,
                                            f"animation_{context.animator.planet_name}_{np.round(context.animator.closest_wavelength.to(u.um).value, 3)}um_{datetime.now().strftime('%Y%m%d_%H%M%S.%f')}.gif",
                                            300):
            data_generator = DataGenerator(context, GenerationMode.data)
            return data_generator.generate_data()
