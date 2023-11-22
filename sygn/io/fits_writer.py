from datetime import datetime
from pathlib import Path

from astropy.io import fits

from sygn.core.context import Context
from sygn.core.entities.photon_sources.planet import Planet
from sygn.core.entities.photon_sources.star import Star


class FITSWriter():

    @staticmethod
    def _get_fits_header(primary: fits.PrimaryHDU, context: Context) -> fits.header.Header:
        """Return the FITS file header containing the information about the simulation and the photon_sources.

        :param primary: The primary HDU object
        :param context: The contexts object
        :param index_target_system: The index of the target system
        :return: The header
        """
        header = primary.header
        header['SYGN_GRID_SIZE'] = context.settings.grid_size
        header['SYGN_TIME_STEP'] = str(context.settings.time_step)
        header['SYGN_STELLAR_LEAKAGE'] = context.settings.noise_contributions.stellar_leakage
        header['SYGN_LOCAL_ZODI_LEAKAGE'] = context.settings.noise_contributions.local_zodi_leakage
        header['SYGN_EXOZODI_LEAKAGE'] = context.settings.noise_contributions.exozodi_leakage
        header[
            'SYGN_FIBER_INJECTION_VARIABILITY'] = context.settings.noise_contributions.fiber_injection_variability
        header[
            'SYGN_OPD_VARIABILITY_APPLY'] = context.settings.noise_contributions.optical_path_difference_variability.apply
        header[
            'SYGN_OPD_VARIABILITY_POWER_LAW_EXPONENT'] = context.settings.noise_contributions.optical_path_difference_variability.power_law_exponent
        header['SYGN_OPD_VARIABILITY_RMS'] = str(
            context.settings.noise_contributions.optical_path_difference_variability.rms)
        header['SYGN_ADJUST_BASELINE_TO_HABITABLE_ZONE'] = context.mission.adjust_baseline_to_habitable_zone
        header['SYGN_INTEGRATION_TIME'] = str(context.mission.integration_time)
        header['SYGN_OPTIMIZED_WAVELENGTH'] = str(context.mission.optimized_wavelength)
        header['SYGN_ARRAY_CONFIGURATION_TYPE'] = context.observatory.array_configuration.type
        header['SYGN_BASELINE_MAXIMUM'] = str(
            context.observatory.array_configuration.baseline_maximum)
        header['SYGN_BASELINE_MINIMUM'] = str(
            context.observatory.array_configuration.baseline_minimum)
        header['SYGN_BASELINE_RATIO'] = context.observatory.array_configuration.baseline_ratio
        header['SYGN_MODULATION_PERIOD'] = str(
            context.observatory.array_configuration.modulation_period)
        header[
            'SYGN_BEAM_COMBINATION_SCHEME'] = context.observatory.beam_combination_scheme.type.value
        header['SYGN_APERTURE_DIAMETER'] = str(
            context.observatory.instrument_parameters.aperture_diameter)
        header[
            'SYGN_SPECTRAL_RESOLVING_POWER'] = context.observatory.instrument_parameters.spectral_resolving_power
        header['SYGN_WAVELENGTH_RANGE_LOWER_LIMIT'] = str(
            context.observatory.instrument_parameters.wavelength_range_lower_limit)
        header['SYGN_WAVELENGTH_RANGE_UPPER_LIMIT'] = str(
            context.observatory.instrument_parameters.wavelength_range_upper_limit)
        header[
            'SYGN_UNPERTURBED_INSTRUMENT_THROUGHPUT'] = context.observatory.instrument_parameters.unperturbed_instrument_throughput
        for source in context.target_specific_photon_sources:
            if isinstance(source, Planet):
                header[f'SYGN_PLANET_{source.name}_MASS'] = str(source.mass)
                header[f'SYGN_PLANET_{source.name}_RADIUS'] = str(source.radius)
                header[f'SYGN_PLANET_{source.name}_TEMPERATURE'] = str(source.temperature)
                header[f'SYGN_PLANET_{source.name}_SEMI_MAJOR_AXIS'] = str(source.semi_major_axis)
                header[f'SYGN_PLANET_{source.name}_ECCENTRICITY'] = source.eccentricity
                header[f'SYGN_PLANET_{source.name}_INCLINATION'] = str(source.inclination)
                header[f'SYGN_PLANET_{source.name}_RAAN'] = str(source.raan)
                header[f'SYGN_PLANET_{source.name}_ARG_OF_PERIAPSIS'] = str(source.argument_of_periapsis)
                header[f'SYGN_PLANET_{source.name}_TRUE_ANOMALY'] = str(source.true_anomaly)
            if isinstance(source, Star):
                header['SYGN_STAR_NAME'] = source.name
                header['SYGN_STAR_DISTANCE'] = str(source.distance)
                header['SYGN_STAR_MASS'] = str(source.mass)
                header['SYGN_STAR_RADIUS'] = str(source.radius)
                header['SYGN_STAR_TEMPERATURE'] = str(source.temperature)
                header['SYGN_STAR_LUMINOSITY'] = str(source.luminosity)
                header['SYGN_STAR_ZODI_LEVEL'] = source.zodi_level
        return header

    @staticmethod
    def write_fits(output_path: Path, context: Context):
        """Write the differential photon counts to a FITS file.

        :param output_path: The output path of the FITS file
        :param simulation: The simulation object
        :param differential_photon_counts: The differential photon counts
        """
        hdu_list = []
        primary = fits.PrimaryHDU()
        header = FITSWriter._get_fits_header(primary, context)
        hdu_list.append(primary)
        for data in context.data:
            hdu = fits.ImageHDU(data)
            hdu_list.append(hdu)
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(f'data_{datetime.now().strftime("%Y%m%d_%H%M%S.%f")}.fits')
