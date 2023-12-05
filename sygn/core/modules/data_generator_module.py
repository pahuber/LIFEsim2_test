from sygn.core.context import Context
from sygn.core.entities.photon_sources.star import Star
from sygn.core.modules.base_module import BaseModule
from sygn.core.processing.data_generation import DataGenerator, GenerationMode


class DataGeneratorModule(BaseModule):
    """Class representation of the data generator modules.
    """

    def __init__(self):
        """Constructor method.

        :param path_to_config_file: Path to the config file
        """

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
        star = [source for source in context.photon_sources if isinstance(source, Star)][0]
        context.observatory.array_configuration.set_optimal_baseline(context.mission.optimized_wavelength,
                                                                     star.habitable_zone_central_angular_radius)
        data_generator = DataGenerator(context, GenerationMode.data)
        context.data = data_generator.generate_data()
        return context
