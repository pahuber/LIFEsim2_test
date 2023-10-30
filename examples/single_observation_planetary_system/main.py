from astropy import units as u

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
# simulation.animate(output_path='.', source_name='Earth', wavelength=10 * u.um, differential_intensity_response_index=0,
#                    photon_counts_limits=100, collector_position_limits=50, image_vmin=-8, image_vmax=8)

# Run simulation
simulation.run()

# Plot the photon count time series for the total signal and for Earth at 10 um
simulation.output.plot_photon_count_time_series('Earth', 10 * u.um)

# SNR
# photon_counts_signal = simulation.output.get_photon_count_time_series_for_source('Earth', 10 * u.um)
# photon_counts_noise = simulation.output.get_photon_count_time_series_for_source('Sun', 10 * u.um)
#
# signal = []
# noise = []
# for wavelength in simulation.observation.observatory.instrument_parameters.wavelength_bin_centers:
#     signal.append(simulation.output.photon_counts_per_wavelength_bin['Earth'][wavelength].value)
#     noise.append(simulation.output.photon_counts_per_wavelength_bin['Sun'][wavelength].value)
# plt.step(range(len(simulation.observation.observatory.instrument_parameters.wavelength_bin_centers)),
#          signal, label="Earth")
# plt.step(range(len(simulation.observation.observatory.instrument_parameters.wavelength_bin_centers)),
#          noise, label="Sun")
# plt.step(range(len(simulation.observation.observatory.instrument_parameters.wavelength_bin_centers)),
#          np.array(signal) / np.array(noise), label="SNR")
# plt.legend()
#
# plt.show()
