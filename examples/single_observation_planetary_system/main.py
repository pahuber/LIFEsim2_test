from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from sygn.core.modules.config_loader_module import ConfigLoaderModule
from sygn.core.modules.data_generator_module import DataGeneratorModule
from sygn.core.modules.target_loader_module import TargetLoaderModule
from sygn.core.pipeline import Pipeline

# Specify paths
path_to_config_file = Path(r'config.yaml')
path_to_context_file = Path(r'planetary_system.yaml')

# Instantiate pipeline
pipeline = Pipeline()

# Load configurations
module = ConfigLoaderModule(path_to_config_file)
pipeline.add_module(module)

# Load target
module = TargetLoaderModule(path_to_context_file)
pipeline.add_module(module)

# Generate data
module = DataGeneratorModule()
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

plt.imshow(np.swapaxes(pipeline._modules[2].differential_photon_counts, axis1=0, axis2=1), cmap='Greys')
plt.title('Differential Photon Counts')
plt.ylabel('Spectral Channel')
plt.xlabel('Time')
plt.colorbar()
plt.tight_layout()
plt.show()

a = 0
# Save synthetic data to FITS file
# pipeline.save_data_to_fits('.')
