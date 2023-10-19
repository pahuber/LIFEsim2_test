import astropy


class Animator():
    """Class representation of an animator to help create animations.
    """

    def __init__(self,
                 output_path: str,
                 source_name: str,
                 closest_wavelength: astropy.units.Quantity,
                 image_vmin: float,
                 image_vmax: float):
        """Constructor method.

        :param output_path: Output path for the animation file
        :param source_name: Name of the source for which the animation should be made
        :param wavelength: Wavelength at which the animation should be made
        :param image_vmin: Minimum value of the colormap
        :param image_vmax: Maximum value of the colormap
        """
        self.output_path = output_path
        self.source_name = source_name
        self.closest_wavelength = closest_wavelength
        self.image_vmin = image_vmin
        self.image_vmax = image_vmax
