import os
from datetime import datetime
from pathlib import Path

from astropy.io import fits

from sygn.core.context import Context
from sygn.core.entities.photon_sources.exozodi import Exozodi
from sygn.core.entities.photon_sources.local_zodi import LocalZodi
from sygn.core.entities.photon_sources.planet import Planet
from sygn.core.entities.photon_sources.star import Star
from sygn.util.helpers import FITSDataType


class FITSWriter():
    """Class representation of the FITS writer.
    """

    @staticmethod
    def _get_fits_header(primary: fits.PrimaryHDU, context: Context, data_type: FITSDataType) -> fits.header.Header:
        """Return the FITS file header containing the information about the simulation and the photon_sources.

        :param primary: The primary HDU object
        :param context: The contexts object
        :param data_type: The data type to be written to FITS
        :return: The header
        """
        header = primary.header

        # The following properties are relevant for both data types
        header['HIERARCH SYGN_GRID_SIZE'] = context.settings.grid_size
        header['HIERARCH SYGN_TIME_STEP'] = str(context.settings.time_step)
        header['HIERARCH SYGN_ADJUST_BASELINE_TO_HABITABLE_ZONE'] = context.mission.adjust_baseline_to_habitable_zone
        header['HIERARCH SYGN_INTEGRATION_TIME'] = str(context.mission.integration_time)
        header['HIERARCH SYGN_OPTIMIZED_WAVELENGTH'] = str(context.mission.optimized_wavelength)
        header['HIERARCH SYGN_ARRAY_CONFIGURATION_TYPE'] = context.observatory.array_configuration.type
        header['HIERARCH SYGN_BASELINE_MAXIMUM'] = str(
            context.observatory.array_configuration.baseline_maximum)
        header['HIERARCH SYGN_BASELINE_MINIMUM'] = str(
            context.observatory.array_configuration.baseline_minimum)
        header['HIERARCH SYGN_BASELINE_RATIO'] = context.observatory.array_configuration.baseline_ratio
        header['HIERARCH SYGN_MODULATION_PERIOD'] = str(
            context.observatory.array_configuration.modulation_period)
        header[
            'HIERARCH SYGN_BEAM_COMBINATION_SCHEME'] = context.observatory.beam_combination_scheme.type.value
        header[
            'HIERARCH SYGN_SPECTRAL_RESOLVING_POWER'] = context.observatory.instrument_parameters.spectral_resolving_power
        header['HIERARCH SYGN_WAVELENGTH_RANGE_LOWER_LIMIT'] = str(
            context.observatory.instrument_parameters.wavelength_range_lower_limit)
        header['HIERARCH SYGN_WAVELENGTH_RANGE_UPPER_LIMIT'] = str(
            context.observatory.instrument_parameters.wavelength_range_upper_limit)

        for source in context.photon_sources:
            if isinstance(source, Planet):
                header[f'HIERARCH SYGN_PLANET_{source.name}_MASS'] = str(source.mass)
                header[f'HIERARCH SYGN_PLANET_{source.name}_RADIUS'] = str(source.radius)
                header[f'HIERARCH SYGN_PLANET_{source.name}_TEMPERATURE'] = str(source.temperature)
                header[f'HIERARCH SYGN_PLANET_{source.name}_SEMI_MAJOR_AXIS'] = str(source.semi_major_axis)
                header[f'HIERARCH SYGN_PLANET_{source.name}_ECCENTRICITY'] = source.eccentricity
                header[f'HIERARCH SYGN_PLANET_{source.name}_INCLINATION'] = str(source.inclination)
                header[f'HIERARCH SYGN_PLANET_{source.name}_RAAN'] = str(source.raan)
                header[f'HIERARCH SYGN_PLANET_{source.name}_ARG_OF_PERIAPSIS'] = str(source.argument_of_periapsis)
                header[f'HIERARCH SYGN_PLANET_{source.name}_TRUE_ANOMALY'] = str(source.true_anomaly)
            if isinstance(source, Star):
                header['HIERARCH SYGN_STAR_NAME'] = source.name
                header['HIERARCH SYGN_STAR_DISTANCE'] = str(source.distance)
                header['HIERARCH SYGN_STAR_MASS'] = str(source.mass)
                header['HIERARCH SYGN_STAR_RADIUS'] = str(source.radius)
                header['HIERARCH SYGN_STAR_TEMPERATURE'] = str(source.temperature)
                header['HIERARCH SYGN_STAR_LUMINOSITY'] = str(source.luminosity)
                header['HIERARCH SYGN_STAR_RIGHT_ASCENSION'] = str(source.right_ascension)
                header['HIERARCH SYGN_STAR_DECLINATION'] = str(source.declination)
            if isinstance(source, Exozodi) and data_type == FITSDataType.SyntheticMeasurement:
                header['HIERARCH SYGN_EXOZODI_LEVEL'] = str(source.level)
                header['HIERARCH SYGN_EXOZODI_INCLINATION'] = str(source.inclincation)
            if isinstance(source, LocalZodi) and data_type == FITSDataType.SyntheticMeasurement:
                header['HIERARCH SYGN_LOCAL_ZODI'] = True

        if data_type == FITSDataType.SyntheticMeasurement:
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
        return header

    @staticmethod
    def write_fits(output_path: Path, context: Context, data_type: FITSDataType):
        """Write the differential photon counts to a FITS file.

        :param output_path: The output path of the FITS file
        :param context: The context
        :param data_type: The data type to be written to FITS
        """
        primary = fits.PrimaryHDU()
        header = FITSWriter._get_fits_header(primary, context, data_type)
        if data_type == FITSDataType.SyntheticMeasurement:
            hdu_list = []
            hdu_list.append(primary)
            for data_per_output in context.data:
                hdu = fits.ImageHDU(data_per_output)
                hdu_list.append(hdu)
            hdul = fits.HDUList(hdu_list)
            hdul.writeto(output_path.joinpath(f'data_{datetime.now().strftime("%Y%m%d_%H%M%S.%f")}.fits'))
        elif data_type == FITSDataType.Template:
            # Create folder
            folder_name = f'templates_{datetime.now().strftime("%Y%m%d_%H%M%S.%f")}'
            os.makedirs(output_path.joinpath(folder_name))

            for index_template, template in enumerate(context.templates):
                hdu_list = []
                hdu_list.append(primary)
                for data_per_output in template:
                    hdu = fits.ImageHDU(data_per_output)
                    hdu_list.append(hdu)
                hdul = fits.HDUList(hdu_list)
                hdul.writeto(output_path.joinpath(folder_name).joinpath(
                    f'template_{index_template}_{datetime.now().strftime("%Y%m%d_%H%M%S.%f")}.fits'))
