from lifesim2.core.data_generation.data_generator import DataGenerator
from lifesim2.core.data_processing.data_processor import DataProcessor
from lifesim2.core.simulation.simulation import Simulation, SimulationMode
from lifesim2.io.data_type import DataType

# Specify paths
path_to_config_file = r'C:\Users\huber\Desktop\LIFEsim2\examples\single_observation_planetary_system\config.yaml'
path_to_data_file = r'C:\Users\huber\Desktop\LIFEsim2\examples\single_observation_planetary_system\planetary_system.yaml'
output_path = r'C:\Users\huber\Desktop\LIFEsim2\examples\single_observation_planetary_system'

# Create simulation object and load configuration and sources
simulation = Simulation()
simulation.load_config(path_to_config_file=path_to_config_file)
simulation.load_sources(data_type=DataType.PLANETARY_SYSTEM_CONFIGURATION, path_to_data_file=path_to_data_file)
# simulation.add_animator(output_path='.', source_name='Earth', wavelength=10 * u.um,
#                         differential_intensity_response_index=0, photon_counts_limits=150, collector_position_limits=50,
#                         image_vmin=-10, image_vmax=10)

# Generate the photon count time series for a single observation
data_generator = DataGenerator(simulation=simulation, simulation_mode=SimulationMode.SINGLE_OBSERVATION)
data_generator.run()
data_generator.save_to_fits(output_path=output_path)

# Create data processor object for processing of synthetic data
data_processor = DataProcessor(photon_count_time_series=data_generator.output.photon_count_time_series)
