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
simulation.output.plot_photon_count_time_series(['Earth'], 10 * u.um, plot_total_counts=True)
