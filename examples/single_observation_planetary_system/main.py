from astropy import units as u
from matplotlib import pyplot as plt

from lifesim2.core.simulation import Simulation, SimulationMode
from lifesim2.io.data_type import DataType

# Specify paths
path_to_config_file = r'C:\Users\huber\Desktop\LIFEsim2\examples\single_observation_planetary_system\config.yaml'
path_to_data_file = r'C:\Users\huber\Desktop\LIFEsim2\examples\single_observation_planetary_system\planetary_system.yaml'

# Create simulation object
simulation = Simulation(mode=SimulationMode.SINGLE_OBSERVATION)

# Load simulation configurations from configuration file
simulation.load_config(path_to_config_file=path_to_config_file)

# Import data
simulation.import_data(type=DataType.PLANETARY_SYSTEM_CONFIGURATION, path_to_data_file=path_to_data_file)

# Create animations during simulation
simulation.animate(output_path='.', source_name='Earth', wavelength=6 * u.um, differential_intensity_response_index=0,
                   image_vmin=-1, image_vmax=1, photon_rate_limits=0.2, collector_position_limits=100)

# Run simulation
simulation.run()

# Extract photon rate time series and plot them
photon_rate_time_series = simulation.output.photon_rate_time_series
photon_rate_time_series_total = simulation.output.photon_rate_time_series_total

wavelength_bin_centers = simulation.observation.observatory.instrument_parameters.wavelength_bin_centers
# for index0, source in enumerate(photon_rate_time_series.keys()):
#     for index, wavelength in enumerate(photon_rate_time_series[source].keys()):
#         if u.Quantity(wavelength) >= 10 * u.um and u.Quantity(wavelength) <= 11 * u.um:
#             plt.plot(photon_rate_time_series[source][wavelength][0], label=f'{source}, {str(wavelength)}')
for index0, wavelength in enumerate(photon_rate_time_series_total.keys()):
    if u.Quantity(wavelength) >= 10 * u.um and u.Quantity(wavelength) <= 11 * u.um:
        plt.plot(photon_rate_time_series_total[wavelength][0], label=f'total, {str(wavelength)}')
        # plt.plot(photon_rate_time_series['Earth'][wavelength][0], label=str(wavelength))
plt.legend()
plt.show()
