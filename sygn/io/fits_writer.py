import os
from datetime import datetime
from itertools import product
from pathlib import Path

from astropy.io import fits

from sygn.core.context import Context
from sygn.core.entities.photon_sources.exozodi import Exozodi
from sygn.core.entities.photon_sources.local_zodi import LocalZodi
from sygn.core.entities.photon_sources.planet import Planet
from sygn.util.helpers import FITSReadWriteType


class FITSWriter():
    """Class representation of the FITS writer.
    """

    @staticmethod
    def _get_fits_header(primary: fits.PrimaryHDU, context: Context,
                         data_type: FITSReadWriteType, index_x: int = None, index_y: int = None) -> fits.header.Header:
        """Return the FITS file header containing the information about the simulation and the photon_sources.

        :param primary: The primary HDU object
        :param context: The contexts object
        :param data_type: The data type to be written to FITS
        :param index_template: The index of the template
        :return: The header
        """
        header = primary.header

        # The following properties are relevant for both data types
        header['HIERARCH SYGN_GRID_SIZE'] = context.settings.grid_size
        header['HIERARCH SYGN_TIME_STEPS'] = str(context.settings.time_steps)
        header['HIERARCH SYGN_PLANET_ORBITAL_MOTION'] = context.settings.planet_orbital_motion
        header['HIERARCH SYGN_INTEGRATION_TIME'] = str(context.mission.integration_time)
        header['HIERARCH SYGN_MODULATION_PERIOD'] = str(context.mission.modulation_period)
        header['HIERARCH SYGN_BASELINE_RATIO'] = context.mission.baseline_ratio
        header['HIERARCH SYGN_BASELINE_MAXIMUM'] = str(context.mission.baseline_maximum)
        header['HIERARCH SYGN_BASELINE_MINIMUM'] = str(context.mission.baseline_minimum)
        header['HIERARCH SYGN_OPTIMIZED_DIFFERENTIAL_OUTPUT'] = str(context.mission.optimized_differential_output)
        header['HIERARCH SYGN_OPTIMIZED_STAR_SEPARATION'] = str(context.mission.optimized_star_separation)
        header['HIERARCH SYGN_OPTIMIZED_WAVELENGTH'] = str(context.mission.optimized_wavelength)
        header['HIERARCH SYGN_ARRAY_CONFIGURATION_TYPE'] = context.observatory.array_configuration.type.value
        header['HIERARCH SYGN_BEAM_COMBINATION_SCHEME'] = context.observatory.beam_combination_scheme.type
        header[
            'HIERARCH SYGN_SPECTRAL_RESOLVING_POWER'] = context.observatory.instrument_parameters.spectral_resolving_power
        header['HIERARCH SYGN_WAVELENGTH_RANGE_LOWER_LIMIT'] = str(
            context.observatory.instrument_parameters.wavelength_range_lower_limit)
        header['HIERARCH SYGN_WAVELENGTH_RANGE_UPPER_LIMIT'] = str(
            context.observatory.instrument_parameters.wavelength_range_upper_limit)

        # The star is added separately outside of the source loop, since it should be written to the file (to recover
        # its properties when the file is read again) even if it is not present in the list of photon sources, i.e. if
        # there is no stellar leakage
        header['HIERARCH SYGN_STAR_NAME'] = context.star.name
        header['HIERARCH SYGN_STAR_DISTANCE'] = str(context.star.distance)
        header['HIERARCH SYGN_STAR_MASS'] = str(context.star.mass)
        header['HIERARCH SYGN_STAR_RADIUS'] = str(context.star.radius)
        header['HIERARCH SYGN_STAR_TEMPERATURE'] = str(context.star.temperature)
        header['HIERARCH SYGN_STAR_LUMINOSITY'] = str(context.star.luminosity)
        header['HIERARCH SYGN_STAR_RIGHT_ASCENSION'] = str(context.star.right_ascension)
        header['HIERARCH SYGN_STAR_DECLINATION'] = str(context.star.declination)

        for source in context.photon_sources:
            if isinstance(source, Planet):
                header[f'HIERARCH SYGN_PLANET_{source.name}_MASS'] = str(source.mass)
                header[f'HIERARCH SYGN_PLANET_{source.name}_RADIUS'] = str(source.radius)
                header[f'HIERARCH SYGN_PLANET_{source.name}_TEMPERATURE'] = str(source.temperature)
                header[f'HIERARCH SYGN_PLANET_{source.name}_SEMI_MAJOR_AXIS'] = str(source.semi_major_axis)
                header[f'HIERARCH SYGN_PLANET_{source.name}_ECCENTRICITY'] = source.eccentricity
                header[f'HIERARCH SYGN_PLANET_{source.name}_INCLINATION'] = str(source.inclination)
                header[f'HIERARCH SYGN_PLANET_{source.name}_RAAN'] = str(source.raan)
                header[f'HIERARCH SYGN_PLANET_{source.name}_ARGUMENT_OF_PERIAPSIS'] = str(source.argument_of_periapsis)
                header[f'HIERARCH SYGN_PLANET_{source.name}_TRUE_ANOMALY'] = str(source.true_anomaly)
            if isinstance(source, Exozodi) and data_type == FITSReadWriteType.SyntheticMeasurement:
                header['HIERARCH SYGN_EXOZODI_LEVEL'] = str(source.level)
                header['HIERARCH SYGN_EXOZODI_INCLINATION'] = str(source.inclination)
            if isinstance(source, LocalZodi) and data_type == FITSReadWriteType.SyntheticMeasurement:
                header['HIERARCH SYGN_LOCAL_ZODI'] = True

        if data_type == FITSReadWriteType.SyntheticMeasurement:
            header['HIERARCH SYGN_STELLAR_LEAKAGE'] = context.settings.noise_contributions.stellar_leakage
            header['HIERARCH SYGN_LOCAL_ZODI_LEAKAGE'] = context.settings.noise_contributions.local_zodi_leakage
            header['HIERARCH SYGN_EXOZODI_LEAKAGE'] = context.settings.noise_contributions.exozodi_leakage
            header[
                'HIERARCH SYGN_FIBER_INJECTION_VARIABILITY'] = context.settings.noise_contributions.fiber_injection_variability
            header[
                'HIERARCH SYGN_OPD_VARIABILITY_APPLY'] = context.settings.noise_contributions.optical_path_difference_variability.apply
            header[
                'HIERARCH SYGN_OPD_VARIABILITY_POWER_LAW_EXPONENT'] = context.settings.noise_contributions.optical_path_difference_variability.power_law_exponent
            header['HIERARCH SYGN_OPD_VARIABILITY_RMS'] = str(
                context.settings.noise_contributions.optical_path_difference_variability.rms)
            header['HIERARCH SYGN_APERTURE_DIAMETER'] = str(
                context.observatory.instrument_parameters.aperture_diameter)
            header[
                'HIERARCH SYGN_UNPERTURBED_INSTRUMENT_THROUGHPUT'] = context.observatory.instrument_parameters.unperturbed_instrument_throughput

        if data_type == FITSReadWriteType.Template:
            header[f'HIERARCH SYGN_INDEX_X'] = index_x
            header[f'HIERARCH SYGN_INDEX_Y'] = index_y
        return header

    @staticmethod
    def write_fits(output_path: Path, context: Context, data_type: FITSReadWriteType):
        """Write the differential photon counts to a FITS file.

        :param output_path: The output path of the FITS file
        :param context: The context
        :param data_type: The data type to be written to FITS
        """
        primary = fits.PrimaryHDU()
        if data_type == FITSReadWriteType.SyntheticMeasurement:
            header = FITSWriter._get_fits_header(primary, context, data_type)
            hdu_list = []
            hdu_list.append(primary)
            for data_per_output in context.signal:
                hdu = fits.ImageHDU(data_per_output)
                hdu_list.append(hdu)
            hdul = fits.HDUList(hdu_list)
            hdul.writeto(output_path.joinpath(f'data_{datetime.now().strftime("%Y%m%d_%H%M%S.%f")}.fits'))
        elif data_type == FITSReadWriteType.Template:
            # Create folder
            folder_name = f'templates_{datetime.now().strftime("%Y%m%d_%H%M%S.%f")}'
            os.makedirs(output_path.joinpath(folder_name))

            for index_x, index_y in product(range(context.settings.grid_size),
                                            range(context.settings.grid_size)):
                template = context.templates[index_x, index_y]
                header = FITSWriter._get_fits_header(primary, context, data_type, index_x, index_y)
                hdu_list = []
                hdu_list.append(primary)
                # Add template data
                for data_per_output in template.signal:
                    hdu = fits.ImageHDU(data_per_output)
                    hdu_list.append(hdu)
                # Add effective areas rms
                for area_per_output in template.effective_area_rms:
                    # table = Table([np.rec.array(area_per_output)], names=['effective_area_rms'], dtype=['float64'])
                    hdu = fits.BinTableHDU.from_columns(
                        [fits.Column(name='effective_area_rms', array=area_per_output, format='D')])
                    hdu_list.append(hdu)
                hdul = fits.HDUList(hdu_list)
                hdul.writeto(output_path.joinpath(folder_name).joinpath(
                    f'template_{datetime.now().strftime("%Y%m%d_%H%M%S.%f")}_{index_x}_{index_y}.fits'))
