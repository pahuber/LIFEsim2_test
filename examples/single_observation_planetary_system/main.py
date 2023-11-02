from astropy import units as u

from lifesim2.core.data_generation.data_generator import DataGenerator
from lifesim2.core.data_processing.data_processor import DataProcessor
from lifesim2.core.simulation.simulation import Simulation, SimulationMode
from lifesim2.io.data_type import DataType

# Specify paths
path_to_config_file = r'C:\Users\huber\Desktop\LIFEsim2\examples\single_observation_planetary_system\config.yaml'
path_to_data_file = r'C:\Users\huber\Desktop\LIFEsim2\examples\single_observation_planetary_system\planetary_system.yaml'

# Create simulation object and load configuration and sources
simulation = Simulation()
simulation.load_config(path_to_config_file=path_to_config_file)
simulation.load_sources(data_type=DataType.PLANETARY_SYSTEM_CONFIGURATION, path_to_data_file=path_to_data_file)
# simulation.add_animator(output_path='.', source_name='Earth', wavelength=10 * u.um,
#                         differential_intensity_response_index=0, photon_counts_limits=150, collector_position_limits=50,
#                         image_vmin=-10, image_vmax=10)

# Create data data_generation and add input data
data_generator = DataGenerator(simulation=simulation, simulation_mode=SimulationMode.SINGLE_OBSERVATION)

# Generate synthetic photon count time series
data_generator.run()

# Create data processor object for processing and analysis of synthetic data
data_processor = DataProcessor(photon_count_time_series=data_generator.output.photon_count_time_series)
data_processor.calibrate_data()

# Plot the photon count time series for the total signal and for Earth at 10 um
data_processor.plot_photon_count_time_series(['Earth'], 10 * u.um, plot_total_counts=True, time_units=u.d)

# Plot the photon statistics-based SNR
data_processor.plot_total_signal_to_noise_ratio_vs_time()
