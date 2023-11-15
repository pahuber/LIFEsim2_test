from pathlib import Path

from sygn.core.pipeline.generator_pipeline import GeneratorPipeline
from sygn.core.module.data_generator.data_generator import DataGenerationMode
from sygn.core.module.data_generator.data_generator_module import DataGeneratorModule
from sygn.core.module.observation.observation_module import ObservationModule
from sygn.core.module.observatory.observatory_module import ObservatoryModule
from sygn.core.module.settings.settings_module import SettingsModule
from sygn.core.module.target_system.data_type import DataType
from sygn.core.module.target_system.target_system_module import TargetSystemModule

# Specify paths
path_to_config_file = Path(r'C:\Users\huber\Desktop\LIFEsim2\examples\single_observation_planetary_system\config.yaml')
path_to_data_file = Path(
    r'C:\Users\huber\Desktop\LIFEsim2\examples\single_observation_planetary_system\planetary_system.yaml')

# Instantiate pipeline
pipeline = GeneratorPipeline()

# Load settings from config file and add settings module to pipeline
module = SettingsModule(path_to_config_file=path_to_config_file)
pipeline.add_module(module)

# Load observation from config file and add observation module to pipeline
module = ObservationModule(path_to_config_file=path_to_config_file)
pipeline.add_module(module)

# Load observatory from config file and add observatory module to pipeline
module = ObservatoryModule(path_to_config_file=path_to_config_file)
pipeline.add_module(module)

# Load sources from file and add sources module to pipeline
module = TargetSystemModule(data_type=DataType.PLANETARY_SYSTEM_CONFIGURATION, path_to_data_file=path_to_data_file)
pipeline.add_module(module)

# # Instantiate animator module and add it to the pipeline
# module = AnimatorModule(output_path='.',
#                         source_name='Earth',
#                         wavelength=10 * astropy.units.um,
#                         differential_intensity_response_index=0,
#                         photon_count_limits=1000,
#                         collector_position_limits=50,
#                         image_vmin=-10,
#                         image_vmax=10)
# pipeline.add_module(module)

# Instantiate data generator module and add it to the pipeline
module = DataGeneratorModule(mode=DataGenerationMode.SINGLE_OBSERVATION)
pipeline.add_module(module)

# Run pipeline
pipeline.run()

# Save synthetic data to FITS file
pipeline.save_data_to_fits('.')
