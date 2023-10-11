from lifesim2.core.data import DataType
from lifesim2.core.simulation import Simulation, SimulationMode

# Specify paths
path_to_config_file = r'C:\Users\huber\Desktop\LIFEsim2\config.yaml'
path_to_data_file = r'C:\Users\huber\Desktop\LIFEsim2\planetary_system.yaml'

# Create simulation object
simulation = Simulation(mode=SimulationMode.SINGLE_OBSERVATION)

# Load simulation configurations from configuration file
simulation.load_config(path_to_config_file=path_to_config_file)

# Import data
simulation.import_data(type=DataType.PLANETARY_SYSTEM_SPECIFICATION, path_to_data_file=path_to_data_file)

# Run simulation
simulation.run()
