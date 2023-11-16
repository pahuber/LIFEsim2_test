from pathlib import Path

from sygn.core.modules.mission_module import MissionModule
from sygn.core.modules.observatory_module import ObservatoryModule
from sygn.core.modules.planetary_system_module import PlanetarySystemModule
from sygn.core.modules.settings_module import SettingsModule
from sygn.core.pipelines.generator_pipeline import GeneratorPipeline
from sygn.io.data_type import DataType

# Specify paths
path_to_config_file = Path(r'C:\Users\huber\Desktop\LIFEsim2\examples\single_observation_planetary_system\config.yaml')
path_to_data_file = Path(
    r'C:\Users\huber\Desktop\LIFEsim2\examples\single_observation_planetary_system\planetary_system.yaml')

# Instantiate generator pipelines
pipeline = GeneratorPipeline()

# Load settings from config file
module = SettingsModule(path_to_config_file=path_to_config_file)
pipeline.add_module(module)

# Load mission from config file
module = MissionModule(path_to_config_file=path_to_config_file)
pipeline.add_module(module)

# Load observatory from config file
module = ObservatoryModule(path_to_config_file=path_to_config_file)
pipeline.add_module(module)

# Load sources from file
module = PlanetarySystemModule(data_type=DataType.PLANETARY_SYSTEM_CONFIGURATION, path_to_data_file=path_to_data_file)
pipeline.add_module(module)

# # Make animation
# modules = AnimatorModule(output_path='.',
#                         source_name='Earth',
#                         wavelength=10 * astropy.units.um,
#                         differential_intensity_response_index=0,
#                         photon_count_limits=1000,
#                         collector_position_limits=50,
#                         image_vmin=-10,
#                         image_vmax=10)
# pipelines.add_module(modules)

# Run pipelines
pipeline.run()

# Save synthetic data to FITS file
pipeline.save_data_to_fits('.')
