import astropy

from lifesim2.core.context import Context
from lifesim2.core.entities.animator import Animator
from lifesim2.core.modules.base_module import BaseModule
from lifesim2.core.modules.data_generator_module import DataGeneratorModule
from lifesim2.util.grid import get_index_of_closest_value


class AnimatorModule(BaseModule):
    def __init__(self,
                 output_path: str,
                 source_name: str,
                 wavelength: astropy.units.Quantity,
                 differential_intensity_response_index: int,
                 image_vmin: float = -1,
                 image_vmax: float = 1,
                 photon_count_limits: float = 0.1,
                 collector_position_limits: float = 50):
        self.output_path = output_path
        self.source_name = source_name
        self.wavelength = wavelength
        self.differential_intensity_response_index = differential_intensity_response_index
        self.image_vmin = image_vmin
        self.image_vmax = image_vmax
        self.photon_count_limits = photon_count_limits
        self.collector_position_limits = collector_position_limits
        self.animator = None
        self.dependencies = [(DataGeneratorModule,)]

    def apply(self, context: Context) -> Context:
        """Apply the module.  Get the wavelength of the list of wavelength centers that matches the closest to the
        wavelength given by the user and initialize the animator object.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        index_closest_wavelength = get_index_of_closest_value(
            context.observatory.instrument_parameters.wavelength_bin_centers,
            self.wavelength)
        closest_wavelength = context.observatory.instrument_parameters.wavelength_bin_centers[
            index_closest_wavelength]
        self.animator = Animator(self.output_path,
                                 self.source_name,
                                 closest_wavelength,
                                 index_closest_wavelength,
                                 self.differential_intensity_response_index,
                                 self.image_vmin,
                                 self.image_vmax,
                                 self.photon_count_limits,
                                 self.collector_position_limits)
        context.animator = self.animator
        return context
