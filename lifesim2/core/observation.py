import astropy.units
from astropy import units as u

from lifesim2.util.grid import get_meshgrid


class Observation():
    """Class representation of a simulated observation. This is the core object of the simulation process."""

    def __init__(self,
                 adjust_baseline_to_habitable_zone: bool,
                 integration_time: astropy.units.Quantity,
                 optimized_wavelength: astropy.units.Quantity,
                 grid_size: int):
        """Constructor method.

        :param adjust_baseline_to_habitable_zone: If true, the baseline is adjusted to the habitable zone, else it is
                                                  adjusted to the planet position.
        :param integration_time: Integration time for an observation
        :param optimized_wavelength: Wavelength for which the baseline is optimized to the habitable zone
        :param grid_size: Grid size for the caluclations
        """
        self.adjust_baseline_to_habitable_zone = adjust_baseline_to_habitable_zone
        self.integration_time = integration_time
        self.optimized_wavelength = optimized_wavelength
        self.grid_size = grid_size
        # TODO: calculate fov
        self.x_sky_coordinates_map, self.y_sky_coordinates_map = get_meshgrid(0.4 * u.arcsec, self.grid_size)
        self.observatory = None
        self.sources = []
