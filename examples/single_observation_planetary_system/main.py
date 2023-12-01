from pathlib import Path

from matplotlib import pyplot as plt

from sygn.core.modules.fits_reader_module import FITSReaderModule
from sygn.core.modules.mlm_extraction_module import MLMExtractionModule
from sygn.core.pipeline import Pipeline
from sygn.util.helpers import FITSDataType

# Create pipeline
pipeline = Pipeline()

# # Load configurations
# module = ConfigLoaderModule(path_to_config_file=Path(r'config.yaml'))
# pipeline.add_module(module)

# # Load target
# module = TargetLoaderModule(path_to_context_file=Path(r'planetary_system.yaml'))
# pipeline.add_module(module)

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

# # Generate synthetic measurements
# module = DataGeneratorModule()
# pipeline.add_module(module)

# # Write synthetic measurements to FITS file
# module = FITSWriterModule(output_path=Path('.'), data_type=FITSDataType.SyntheticMeasurement)
# pipeline.add_module(module)

# Read synthetic measurement
module = FITSReaderModule(input_path=Path('data_20231201_134346.934838.fits'),
                          data_type=FITSDataType.SyntheticMeasurement)
pipeline.add_module(module)

# # Generate Templates
# module = TemplateGeneratorModule()
# pipeline.add_module(module)

# # Write templates to FITS file
# module = FITSWriterModule(output_path=Path('./templates'), data_type=FITSDataType.Template)
# pipeline.add_module(module)

# Read templates
module = FITSReaderModule(input_path=Path('templates'), data_type=FITSDataType.Template)
pipeline.add_module(module)

# Extract flux and position using maximum likelihood method
module = MLMExtractionModule()
pipeline.add_module(module)

# # Fit input spectrum to extracted spectrum
# module = SpectrumFittingModule()
# pipeline.add_module(module)

# Run pipeline
pipeline.run()

# Plot synthetic measurement
plt.imshow(pipeline.get_data()[0], cmap='Greys')
plt.title('Differential Photon Counts')
plt.ylabel('Spectral Channel')
plt.xlabel('Time')
plt.colorbar()
plt.show()

# Plot template
plt.imshow(pipeline._context.templates[0][0], cmap='Greys')
plt.title('Differential Photon Counts Templates')
plt.ylabel('Spectral Channel')
plt.xlabel('Time')
plt.colorbar()
plt.show()
