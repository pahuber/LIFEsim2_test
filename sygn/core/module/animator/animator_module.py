import astropy

from sygn.core.context.base_context import BaseContext
from sygn.core.module.animator.animator import Animator
from sygn.core.module.base_module import BaseModule
from sygn.util.grid import get_index_of_closest_value


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

    def apply(self, context: BaseContext) -> BaseContext:
        """Apply the module.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        closest_wavelength = context.observatory.instrument_parameters.wavelength_bin_centers[
            get_index_of_closest_value(context.observatory.instrument_parameters.wavelength_bin_centers,
                                       self.wavelength)]
        self.animator = Animator(self.output_path,
                                 self.source_name,
                                 closest_wavelength,
                                 self.differential_intensity_response_index,
                                 self.image_vmin,
                                 self.image_vmax,
                                 self.photon_count_limits,
                                 self.collector_position_limits)
        context.animator = self.animator
        return context
