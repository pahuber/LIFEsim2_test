from pathlib import Path

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lifesim2.core.modules.config_loader_module import ConfigLoaderModule
from lifesim2.core.modules.data_generator_module import DataGeneratorModule
from lifesim2.core.modules.fits_reader_module import FITSReaderModule
from lifesim2.core.modules.flux_calibration_module import FluxCalibrationModule
from lifesim2.core.modules.mlm_extraction_module import MLExtractionModule
from lifesim2.core.modules.target_loader_module import TargetLoaderModule
from lifesim2.core.pipeline import Pipeline
from lifesim2.util.helpers import FITSReadWriteType

########################################################################################################################
# Create Pipeline
########################################################################################################################

pipeline = Pipeline()

########################################################################################################################
# Load configurations from file
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
#                          source_name='Earth',
#                          wavelength=10 * astropy.units.um,
#                          differential_intensity_response_index=0,
#                          photon_count_limits=10000,
#                          collector_position_limits=20,
#                          image_vmin=-20,
#                          image_vmax=20)
# pipeline.add_module(modules)

########################################################################################################################
# Generate synthetic measurements
########################################################################################################################

# Generate synthetic measurements
module = DataGeneratorModule()
pipeline.add_module(module)

# Write synthetic measurements to FITS file
# module = FITSWriterModule(output_path=Path('.'), data_type=FITSReadWriteType.SyntheticMeasurement)
# pipeline.add_module(module)

########################################################################################################################
# Or load synthetic measurements
########################################################################################################################

# module = FITSReaderModule(input_path=Path('data_20231220_141250.389684.fits'),
#                           data_type=FITSReadWriteType.SyntheticMeasurement)
# pipeline.add_module(module)

########################################################################################################################
# Generate templates
########################################################################################################################

# # Generate Templates
# module = TemplateGeneratorModule()
# pipeline.add_module(module)
#
# # Write templates to FITS file
# module = FITSWriterModule(output_path=Path('.'), data_type=FITSReadWriteType.Template)
# pipeline.add_module(module)

########################################################################################################################
# Or load templates
########################################################################################################################

module = FITSReaderModule(input_path=Path('templates_20240105_091157.495302'), data_type=FITSReadWriteType.Template)
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
# Run pipeline
########################################################################################################################

pipeline.run()

########################################################################################################################
# Analyze results
########################################################################################################################

signal = pipeline.get_signal()[0]
planet = pipeline.get_planets()[0]
extractions = pipeline.get_extractions()
spectrum = extractions[0].spectrum[0]
spectrum_uncertainties = extractions[0].spectrum_uncertainties[0]
cost_function = extractions[0].cost_function[0]
wavelengths = [round(wavelength, 1) for wavelength in pipeline.get_wavelengths().value]

# Plot synthetic signal
plt.figure()
ax = plt.gca()
im = ax.imshow(signal, cmap='Greys')
plt.title('Differential Photon Counts')
plt.ylabel('Spectral Channel')
plt.xlabel('Time')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.savefig('photon_counts.pdf', bbox_inches='tight')
plt.show()

# Plot cost function
plt.imshow(cost_function, cmap='magma')
plt.title('Cost Function')
plt.colorbar()
plt.show()

# Plot extracted spectrum
spectrum_uncertainties[-1] = 0

plt.scatter(wavelengths, spectrum.value, label='Synthetic Data', color='royalblue')
plt.errorbar(wavelengths, spectrum.value, yerr=spectrum_uncertainties.value, ls='none', color='royalblue')
plt.plot(wavelengths, planet.mean_spectral_flux_density.value, color='k', label='Model')
plt.fill_between(wavelengths,
                 planet.mean_spectral_flux_density.value - spectrum_uncertainties.value,
                 planet.mean_spectral_flux_density.value + spectrum_uncertainties.value,
                 color='black', alpha=0.1, label='1-$\sigma$')
plt.title('Planet Spectrum')
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Flux Density (ph s$^{-1}$ m$^{-2}$ $\mu$m$^{-1}$)')
plt.tight_layout()
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
