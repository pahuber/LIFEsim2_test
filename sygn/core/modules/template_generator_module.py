import numpy as np
from tqdm.contrib.itertools import product

from sygn.core.context import Context
from sygn.core.entities.photon_sources.planet import Planet
from sygn.core.modules.base_module import BaseModule
from sygn.core.modules.config_loader_module import ConfigLoaderModule
from sygn.core.modules.fits_reader_module import FITSReaderModule
from sygn.core.modules.target_loader_module import TargetLoaderModule
from sygn.core.processing.data_generation import DataGenerator, GenerationMode
from sygn.core.template import Template
from sygn.util.helpers import FITSReadWriteType


class TemplateGeneratorModule(BaseModule):
    """Class representation of the template generator module.
    """

    def __init__(self):
        """Constructor method.
        """
        self.dependencies = [(ConfigLoaderModule, TargetLoaderModule),
                             (ConfigLoaderModule, FITSReaderModule, FITSReadWriteType.SyntheticMeasurement),
                             (TargetLoaderModule, FITSReaderModule, FITSReadWriteType.SyntheticMeasurement),
                             (FITSReaderModule, FITSReadWriteType.SyntheticMeasurement)]

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
        context.observatory.set_optimal_baseline(context.star,
                                                 context.mission.optimized_differential_output,
                                                 context.mission.optimized_wavelength,
                                                 context.mission.optimized_star_separation,
                                                 context.mission.baseline_minimum,
                                                 context.mission.baseline_maximum)
        context_template = self._unload_noise_contributions(context)

        context.templates = np.zeros((context.settings.grid_size, context.settings.grid_size), dtype=object)

        for source in context.photon_sources:
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
                        context_template.photon_sources = [source]
                        data_generator = DataGenerator(context_template, GenerationMode.template)
                        signal, effective_area = data_generator.generate_data()
                        units = effective_area.unit

                        # Normalize each wavelength to unit RMS
                        normalization = np.sqrt(np.mean(signal ** 2, axis=2))
                        signal = np.einsum('ijk, ij->ijk', signal, 1 / normalization)

                        # Create template objects
                        template = Template(signal, np.sqrt(np.mean(np.array(effective_area) ** 2, axis=2)) * units,
                                            index_x, index_y)
                        context.templates[index_x, index_y] = template

                else:
                    raise Exception('Template generation including planet orbital motion is not yet supported')

        return context
