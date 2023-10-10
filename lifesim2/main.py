from lifesim2.observation.observation import Observation

# Specify paths
path_to_config_file = r'C:\Users\huber\Desktop\LIFEsim2\config.yaml'

# Initialize observation
observation = Observation(path_to_config_file=path_to_config_file)

# Run simulation of observation
observation.run()
