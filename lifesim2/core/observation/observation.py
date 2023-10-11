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
        self.adjust_baseline_to_habitable_zone = adjust_baseline_to_habitable_zone
        self.integration_time = integration_time
        self.optimized_wavelength = optimized_wavelength
        self.grid_size = grid_size
        self.x_sky_coordinates_map, self.y_sky_coordinates_map = get_meshgrid(0.4 * u.arcsec, self.grid_size)
        self.observatory = None

        # self.path_to_config_file = path_to_config_file
        # self.field_of_view = None
        # self.integration_time = None
        # self.modulation_period = None
        # self.observatory = None
        # self.photon_rate_time_series = []
        # self.time_range = None
        # self.x_sky_coordinates_map = None
        # self.y_sky_coordinates_map = None
        #
        # self.simulation_grid_size = None
        # self.simulation_time_step = None

        # self._set_from_config()

    # def _set_from_config(self):
    #     """Read the configuration file and set the Observation parameters.
    #     """
    #     self._config_dict = ConfigReader(path_to_config_file=self.path_to_config_file).get_config_from_file()
    #     self.field_of_view = 0.4 * u.arcsec  # TODO: calculate fov
    #     self.integration_time = self._config_dict['observation']['integration_time']
    #     self.modulation_period = self._config_dict['observatory']['array_configuration']['modulation_period']
    #     self.simulation_grid_size = int(self._config_dict['simulation']['grid_size'].value)
    #     self.simulation_time_step = self._config_dict['simulation']['time_step']
    #
    #     self.time_range = np.arange(0, self.modulation_period.to(u.s).value,
    #                                 self.simulation_time_step.to(u.s).value) * u.s
    #     self.x_sky_coordinates_map, self.y_sky_coordinates_map = get_meshgrid(self.field_of_view,
    #                                                                           self.simulation_grid_size)
    #
    #     self.create_observatory()
