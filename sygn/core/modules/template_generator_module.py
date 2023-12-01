from tqdm.contrib.itertools import product

from sygn.core.context import Context
from sygn.core.entities.photon_sources.planet import Planet
from sygn.core.processing.data_generation import DataGenerator, GenerationMode


class TemplateGeneratorModule():
    """Class representation of the template generator module.
    """

    def __init__(self):
        """Constructor method.
        """

    def _unload_noise_contributions(self, context) -> Context:
        """Unload all noise contributions by setting the corresponding values to false, since they should not be
        considered for creating the templates.

        :param context: The context
        :return: The context with the updated noise contributions
        """
        context.settings.noise_contributions.stellar_leakage = False
        context.settings.noise_contributions.local_zodi_leakage = False
        context.settings.noise_contributions.exozodi_leakage = False
        context.settings.noise_contributions.fiber_injection_variability = False
        context.settings.noise_contributions.optical_path_difference_variability.apply = False
        return context

    def apply(self, context: Context) -> Context:
        """Calculate the (spectral-temporal) templates for each planet and every possible planet position within the
        grid.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        context.observatory.array_configuration.set_optimal_baseline(context.mission.optimized_wavelength,
                                                                     context.star_habitable_zone_central_angular_radius)

        context_template = self._unload_noise_contributions(context)

        for source in context.target_specific_photon_sources:
            if isinstance(source, Planet):
                if not context.settings.planet_orbital_motion:
                    for index_x, index_y in product(range(context.settings.grid_size),
                                                    range(context.settings.grid_size)):
                        for index_wavelength in range(
                                len(context.observatory.instrument_parameters.wavelength_bin_centers)):
                            # Reset source sky distribution map
                            source.sky_brightness_distribution[0][index_wavelength] *= 0

                            # Set new planet position
                            source.sky_brightness_distribution[0][index_wavelength][index_x][index_y] = \
                                source.mean_spectral_flux_density[index_wavelength]

                        # Run data generator
                        context_template.target_specific_photon_sources = [source]
                        data_generator = DataGenerator(context_template, GenerationMode.template)
                        context.templates.append(data_generator.generate_data())
                else:
                    raise Exception('Template generation including planet orbital motion is not yet supported')
        return context
