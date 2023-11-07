from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.io import fits

from sygn.core.context import Context
from sygn.core.modules.target_system.planet import Planet
from sygn.core.modules.target_system.star import Star


class FITSWriter():

    @staticmethod
    def _get_fits_header(primary: fits.PrimaryHDU, context: Context, index_target_system) -> fits.header.Header:
        """Return the FITS file header containing the information about the simulation and the sources.

        :param primary: The primary HDU object
        :param context: The context object
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
        header['SYGN_ADJUST_BASELINE_TO_HABITABLE_ZONE'] = context.observation.adjust_baseline_to_habitable_zone
        header['SYGN_INTEGRATION_TIME'] = str(context.observation.integration_time)
        header['SYGN_OPTIMIZED_WAVELENGTH'] = str(context.observation.optimized_wavelength)
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
        for index_source, source in enumerate(context.target_systems[index_target_system].values()):
            if isinstance(source, Planet):
                header[f'SYGN_PLANET{index_source}_NAME'] = source.name
                header[f'SYGN_PLANET{index_source}_MASS'] = str(source.mass)
                header[f'SYGN_PLANET{index_source}_RADIUS'] = str(source.radius)
                # TODO: add orbital elements
                header[f'SYGN_PLANET{index_source}_TEMPERATURE'] = str(source.temperature)
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
    def write_fits(output_path: Path, postfix: str, context: Context):
        """Write the differential photon counts to a FITS file.

        :param output_path: The output path of the FITS file
        :param postfix: Postfix that is appended to the output file name
        :param simulation: The simulation object
        :param differential_photon_counts: The differential photon counts
        """
        for index_target_system, target_system in enumerate(context.target_systems):
            hdu_list = []
            primary = fits.PrimaryHDU()
            header = FITSWriter._get_fits_header(primary, context, index_target_system)
            hdu_list.append(primary)
            for index_response in context.differential_photon_counts_list[index_target_system].keys():
                differential_photon_counts_list = list(
                    context.differential_photon_counts_list[index_target_system][index_response].values())
                differential_photon_counts_array = np.array(differential_photon_counts_list)
                hdu = fits.ImageHDU(differential_photon_counts_array)
                hdu_list.append(hdu)
            hdul = fits.HDUList(hdu_list)
            hdul.writeto(f'differential_photon_counts_{datetime.now().strftime("%Y%m%d_%H%M%S.%f")}{postfix}.fits')
