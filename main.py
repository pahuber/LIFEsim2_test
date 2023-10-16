from matplotlib import pyplot as plt

from lifesim2.core.simulation import Simulation, SimulationMode
from lifesim2.io.data_type import DataType

# Specify paths
path_to_config_file = r'C:\Users\huber\Desktop\LIFEsim2\config.yaml'
path_to_data_file = r'C:\Users\huber\Desktop\LIFEsim2\planetary_system.yaml'

# Create simulation object
simulation = Simulation(mode=SimulationMode.SINGLE_OBSERVATION)

# Load simulation configurations from configuration file
simulation.load_config(path_to_config_file=path_to_config_file)

# Import data
simulation.import_data(type=DataType.PLANETARY_SYSTEM_CONFIGURATION, path_to_data_file=path_to_data_file)

# Run simulation
simulation.run()

# Extract photon rate time series and plot them
photon_rate_time_series = simulation.output.photon_rate_time_series
for wavelength in photon_rate_time_series.keys():
    plt.plot(photon_rate_time_series[wavelength][0], label=str(wavelength))
plt.legend()
plt.show()
