from astropy import units as u

from lifesim2.core.data_processing.data_processor import DataProcessor, DataProcessingMode
from lifesim2.core.simulation.simulation import Simulation, SimulationMode
from lifesim2.io.data_type import DataType

# Specify paths
path_to_config_file = r'C:\Users\huber\Desktop\LIFEsim2\examples\single_observation_planetary_system\config.yaml'
path_to_data_file = r'C:\Users\huber\Desktop\LIFEsim2\examples\single_observation_planetary_system\planetary_system.yaml'

# Create simulation object
simulation = Simulation(mode=SimulationMode.SINGLE_OBSERVATION)

# Load simulation configurations from configuration file
simulation.load_config(path_to_config_file=path_to_config_file)

# Import data
simulation.load_sources(type=DataType.PLANETARY_SYSTEM_CONFIGURATION, path_to_data_file=path_to_data_file)

# Create animations during simulation
# simulation.animate(output_path='.', source_name='Earth', wavelength=10 * u.um, differential_intensity_response_index=0,
#                    photon_counts_limits=150, collector_position_limits=50, image_vmin=-10, image_vmax=10)

# Run simulation
simulation.run()

# Create data processor object for processing and analysis of synthetic data
data_processor = DataProcessor(simulation)

data_processor.get_snr_per_bin(mode=DataProcessingMode.PHOTON_STATISTICS)
data_processor.get_snr_per_bin(mode=DataProcessingMode.EXTRACTION)
data_processor.get_total_snr(mode=DataProcessingMode.PHOTON_STATISTICS)
data_processor.get_recovered_spectrum(mode=DataProcessingMode.EXTRACTION)

# Plot the photon count time series for the total signal and for Earth at 10 um
data_processor.plot_photon_count_time_series(['Earth'], 10 * u.um, plot_total_counts=True, time_units=u.d)

# Plot the photon statistics-based SNR
data_processor.plot_total_signal_to_noise_ratio_vs_time()
