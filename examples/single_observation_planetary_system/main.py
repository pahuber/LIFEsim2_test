from pathlib import Path

from matplotlib import pyplot as plt

from sygn.core.modules.config_loader_module import ConfigLoaderModule
from sygn.core.modules.data_generator_module import DataGeneratorModule
from sygn.core.modules.fits_writer_module import FITSWriterModule
from sygn.core.modules.target_loader_module import TargetLoaderModule
from sygn.core.modules.template_generator_module import TemplateGeneratorModule
from sygn.core.pipeline import Pipeline
from sygn.util.helpers import FITSDataType

# Create pipeline
pipeline = Pipeline()

# Load configurations
module = ConfigLoaderModule(path_to_config_file=Path(r'config.yaml'))
pipeline.add_module(module)

# Load target
module = TargetLoaderModule(path_to_context_file=Path(r'planetary_system.yaml'))
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

# Generate data
module = DataGeneratorModule()
pipeline.add_module(module)

# Write data to FITS file
module = FITSWriterModule(output_path=Path('.'), data_type=FITSDataType.SyntheticData)
pipeline.add_module(module)

# Generate/Load Data Templates
module = TemplateGeneratorModule()
pipeline.add_module(module)

# Write templates to FITS file
module = FITSWriterModule(output_path=Path('./templates'), data_type=FITSDataType.Template)
pipeline.add_module(module)
#
# # Extract flux and position using cross correlation
# module = XCorExtractionModule()
# pipeline.add_module(module)
#
# # Fit input spectrum to extracted spectrum
# module = SpectrumFittingModule()
# pipeline.add_module(module)

# Run pipeline
pipeline.run()

# Plot output data
plt.imshow(pipeline.get_data()[0], cmap='Greys')
plt.title('Differential Photon Counts')
plt.ylabel('Spectral Channel')
plt.xlabel('Time')
plt.colorbar()
plt.show()

plt.imshow(pipeline._context.templates[0][0], cmap='Greys')
plt.title('Differential Photon Counts Templates')
plt.ylabel('Spectral Channel')
plt.xlabel('Time')
plt.colorbar()
plt.show()
