from pathlib import Path

from matplotlib import pyplot as plt

from sygn.core.entities.photon_sources.planet import Planet
from sygn.core.modules.config_loader_module import ConfigLoaderModule
from sygn.core.modules.data_generator_module import DataGeneratorModule
from sygn.core.modules.fits_reader_module import FITSReaderModule
from sygn.core.modules.fits_writer_module import FITSWriterModule
from sygn.core.modules.flux_calibration_module import FluxCalibrationModule
from sygn.core.modules.mlm_extraction_module import MLExtractionModule
from sygn.core.modules.target_loader_module import TargetLoaderModule
from sygn.core.pipeline import Pipeline
from sygn.util.helpers import FITSDataType

########################################################################################################################
# Create Pipeline
########################################################################################################################

pipeline = Pipeline()

########################################################################################################################
# Generate synthetic measurements
########################################################################################################################

# Load configurations
module = ConfigLoaderModule(path_to_config_file=Path(r'config.yaml'))
pipeline.add_module(module)
#
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
# pipeline.add_module(modules)

# Generate synthetic measurements
module = DataGeneratorModule()
pipeline.add_module(module)

# Write synthetic measurements to FITS file
module = FITSWriterModule(output_path=Path('.'), data_type=FITSDataType.SyntheticMeasurement)
pipeline.add_module(module)

########################################################################################################################
# Or load synthetic measurements
########################################################################################################################

# module = FITSReaderModule(input_path=Path('data_20231208_143856.776090.fits'),
#                           data_type=FITSDataType.SyntheticMeasurement)
# pipeline.add_module(module)

########################################################################################################################
# Generate templates
########################################################################################################################

# # Generate Templates
# module = TemplateGeneratorModule()
# pipeline.add_module(module)
#
# # Write templates to FITS file
# module = FITSWriterModule(output_path=Path('.'), data_type=FITSDataType.Template)
# pipeline.add_module(module)

########################################################################################################################
# Or load templates
########################################################################################################################

module = FITSReaderModule(input_path=Path('templates_20231213_132552.373306'), data_type=FITSDataType.Template)
pipeline.add_module(module)

########################################################################################################################
# Process data
########################################################################################################################

# Extract flux and position using maximum likelihood method
module = MLExtractionModule()
pipeline.add_module(module)

# Calibrate flux
module = FluxCalibrationModule()
pipeline.add_module(module)

# # Fit input spectrum to extracted spectrum
# module = SpectrumFittingModule()
# pipeline.add_module(module)

########################################################################################################################
# Run pipeline and analyse data
########################################################################################################################

# Run pipeline
pipeline.run()

# # Plot synthetic measurement
# plt.imshow(pipeline.get_data()[0], cmap='Greys')
# plt.title('Differential Photon Counts')
# plt.ylabel('Spectral Channel')
# plt.xlabel('Time')
# plt.colorbar()
# plt.show()
#
# # Plot template
# plt.imshow(pipeline._context.templates[0][0], cmap='Greys')
# plt.title('Differential Photon Counts Templates')
# plt.ylabel('Spectral Channel')
# plt.xlabel('Time')
# plt.colorbar()
# plt.show()

# Plot cost function
plt.imshow(pipeline._context.cost_function[:, :, 0], cmap='magma')
plt.title('Cost Function')
plt.colorbar()
plt.show()

# Plot optimized flux
planet = [source for source in pipeline._context.photon_sources if isinstance(source, Planet)][0]
wavelengths = pipeline._context.observatory.instrument_parameters.wavelength_bin_centers.value
wavelengths = [round(wavelength, 1) for wavelength in wavelengths]

# # Plot planet spectrum
plt.scatter(range(len(pipeline._context.flux_density[0])), pipeline._context.flux_density[0].value,
            label='Synthetic Data', color='royalblue')
plt.plot(planet.mean_spectral_flux_density.value, color='k', label='Model')
plt.title('Planet Spectrum')
plt.xlabel('Wavelength ($\mu$m)')
plt.xticks(range(len(wavelengths))[0::5], wavelengths[0::5])
plt.ylabel('Flux Density (ph s$^{-1}$ m$^{-2}$ $\mu$m$^{-1}$)')
plt.legend()
plt.show()

# # Get position
# x = planet.sky_coordinates[0].x
# y = planet.sky_coordinates[0].y
# x_coordinate = x[i1][i2]
# y_coordinate = y[i1][i2]
#
# extent = np.max(x[0]) - np.min(x[0])
# extent_per_pixel = extent / len(x[0])
# uncertainty = extent_per_pixel / 2
#
# print(f'{x_coordinate} +/- {uncertainty}')
# print(f'{y_coordinate} +/- {uncertainty}')
