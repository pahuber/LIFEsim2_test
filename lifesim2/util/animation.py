import astropy
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter

from lifesim2.core.observation import Observation
from lifesim2.io.simulation_output import SimulationOutput


class Animator():
    """Class representation of an animator to help create animations.
    """

    def __init__(self,
                 output_path: str,
                 source_name: str,
                 closest_wavelength: astropy.units.Quantity,
                 differential_intensity_response_index: int,
                 image_vmin: float,
                 image_vmax: float):
        """Constructor method.

        :param output_path: Output path for the animation file
        :param source_name: Name of the source for which the animation should be made
        :param wavelength: Wavelength at which the animation should be made
        :param differential_intensity_response_index: Index specifying which of the differential outputs to animate
        :param image_vmin: Minimum value of the colormap
        :param image_vmax: Maximum value of the colormap
        """
        self.output_path = output_path
        self.source_name = source_name
        self.closest_wavelength = closest_wavelength
        self.differential_intensity_response_index = differential_intensity_response_index
        self.image_vmin = image_vmin
        self.image_vmax = image_vmax
        self.writer = None
        self.images = None
        self.figure = None
        self.photon_rate_list = []
        self.time_index_list = []

    def prepare_animation_writer(self, observation: Observation, time_range: np.ndarray, grid_size: int):
        """Prepare the animation writer.

        :param observation: THe observation object
        :param time_range: The time range
        :param grid_size: The grid size
        """
        cmap = self._get_overlay_cmap()
        self.writer = FFMpegWriter(fps=15)
        self.figure = plt.figure(1)
        gs = self.figure.add_gridspec(2, 2)

        image_intensity_response = self._prepare_differential_intensity_response_plot(gs, observation, grid_size, cmap)
        image_collector_positions = self._prepare_collector_position_image(gs, grid_size)
        image_photon_rates = self._prepare_photon_rates_plot(gs, time_range)

        self.figure.tight_layout()
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0,
                            hspace=0.4)
        plt.close()
        self.images = (image_collector_positions, image_intensity_response, image_photon_rates)

    def update_collector_position(self, time, observation):
        """Get the collector positions and update the frame.

        :param time: The time
        :param observation: The observation object
        """
        x_observatory_coordinates, y_observatory_coordinates = observation.observatory.array_configuration.get_collector_positions(
            time)
        self.images[0].set_data(x_observatory_coordinates.value, y_observatory_coordinates.value)

    def update_differential_intensity_response(self, differential_intensity_responses: np.ndarray):
        """Update the differential_intensity_response frame

        :param differential_intensity_responses: The differential intensity responses
        """
        self.images[1].set_data(differential_intensity_responses[self.differential_intensity_response_index])

    def update_photon_rate(self, output: SimulationOutput, index_time: int):
        """Update the photon rate frame.

        :param output: The output object
        :param index_time: The time index
        :return:
        """
        self.time_index_list.append(index_time)
        self.photon_rate_list.append(
            output.photon_rate_time_series[self.source_name][self.closest_wavelength][
                self.differential_intensity_response_index][index_time])
        self.images[2].set_data(self.time_index_list, self.photon_rate_list)

    def _get_overlay_cmap(self) -> matplotlib.colors.LinearSegmentedColormap:
        """Create a color map to overlay the source on the differential intensity response plot.

        :return: The colormap
        """
        colors = [(0, 1, 1, c) for c in np.linspace(0, 1, 100)]
        return mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=2)

    def _prepare_differential_intensity_response_plot(self,
                                                      gs: matplotlib.gridspec.GridSpec,
                                                      observation: Observation,
                                                      grid_size: int,
                                                      cmap: matplotlib.colors.LinearSegmentedColormap) -> matplotlib.image.AxesImage:
        """Prepare the plot for the differential intensity response.

        :param gs: The GridSpec object
        :param observation: The observation object
        :param grid_size: Tge grid size
        :param cmap: The color map
        :return: The image
        """
        ax1 = self.figure.add_subplot(gs[0, 1])
        image_intensity_response = ax1.imshow(np.zeros((grid_size, grid_size)),
                                              vmin=self.image_vmin,
                                              vmax=self.image_vmax,
                                              cmap='seismic')
        ax1.imshow(observation.sources['Earth'].shape_map, cmap=cmap)

        max = int(np.max(observation.sources[self.source_name].sky_coordinate_maps[0][0, :]).value * 1000)
        labels = np.linspace(-max, max, grid_size // 10 + 1)
        ticks = np.linspace(0, grid_size, grid_size // 10 + 1)

        cb = self.figure.colorbar(image_intensity_response)
        ax1.set_title(f'Differential Intensity Response', fontsize=10)
        ax1.set_xlabel(f'Sky Coordinates (mas)', fontsize=8)
        ax1.set_ylabel(f'Sky Coordinates (mas)', fontsize=8)
        ax1.set_xticks(ticks=ticks, labels=labels)
        ax1.set_yticks(ticks=ticks, labels=labels)
        ax1.tick_params(axis='both', which='major', labelsize=8)
        cb.ax.tick_params(labelsize=8)

        return image_intensity_response

    def _prepare_collector_position_image(self,
                                          gs: matplotlib.gridspec.GridSpec,
                                          grid_size: int) -> matplotlib.image.AxesImage:
        """Prepare the plot for the collector positions.

        :param gs: The GridSpec object
        :param grid_size: The grid size
        :return: The image
        """
        ax3 = self.figure.add_subplot(gs[0, 0])
        image_collector_positions, = ax3.plot([], [], 'bo')

        extent = 50
        labels = np.linspace(-extent, extent, 2 * extent // 10 + 1)
        ticks = np.linspace(0, grid_size, 2 * extent // 10 + 1)

        ax3.set_title(f'Collector Positions', fontsize=10)
        ax3.set_xlabel(f'Position (m)', fontsize=8)
        ax3.set_ylabel(f'Position (m)', fontsize=8)
        ax3.set_xlim(-extent, extent)
        ax3.set_ylim(-extent, extent)
        ax3.set_box_aspect(1)
        ax3.set_aspect('equal')
        ax3.tick_params(axis='both', which='major', labelsize=8)

        return image_collector_positions

    def _prepare_photon_rates_plot(self, gs: matplotlib.gridspec.GridSpec,
                                   time_range: np.ndarray) -> matplotlib.image.AxesImage:
        """Prepare the plot for the photon rates.

        :param gs: The GirdSpec object
        :param time_range: The time range
        :return: The image
        """
        ax2 = self.figure.add_subplot(gs[1, :])
        image_photon_rates, = ax2.plot([], [], 'b-o')

        labels = time_range.to(u.h)
        labels = [int(x.value) for x in labels]

        ax2.set_title(f'Photon Rate', fontsize=10)
        ax2.set_xlabel(f'Time (h)', fontsize=8)
        ax2.set_ylabel(f'Photon Rate (a.u.)', fontsize=8)
        ax2.set_xlim(0, len(time_range))
        ax2.set_ylim(-0.1, 0.1)
        ax2.set_xticklabels(labels=labels)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        ax2.set_box_aspect(0.4)

        return image_photon_rates
