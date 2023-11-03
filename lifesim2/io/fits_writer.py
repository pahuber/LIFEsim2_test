from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.io import fits

from lifesim2.core.simulation.simulation import Simulation
from lifesim2.core.simulation.sources.planet import Planet
from lifesim2.core.simulation.sources.star import Star


class FITSWriter():

    @staticmethod
    def _get_fits_header(primary: fits.PrimaryHDU, simulation: Simulation) -> fits.header.Header:
        """Return the FITS file header containing the information about the simulation and the sources.

        :param primary: The primary HDU object
        :param simulation: THe simulation object
        :return: The header
        """
        header = primary.header
        header['LIFESIM2_GRID_SIZE'] = simulation.config.grid_size
        header['LIFESIM2_TIME_STEP'] = str(simulation.config.time_step)
        header['LIFESIM2_STELLAR_LEAKAGE'] = simulation.config.noise_contributions.stellar_leakage
        header['LIFESIM2_LOCAL_ZODI_LEAKAGE'] = simulation.config.noise_contributions.local_zodi_leakage
        header['LIFESIM2_EXOZODI_LEAKAGE'] = simulation.config.noise_contributions.exozodi_leakage
        header[
            'LIFESIM2_FIBER_INJECTION_VARIABILITY'] = simulation.config.noise_contributions.fiber_injection_variability
        header[
            'LIFESIM2_OPD_VARIABILITY_APPLY'] = simulation.config.noise_contributions.optical_path_difference_variability.apply
        header[
            'LIFESIM2_OPD_VARIABILITY_POWER_LAW_EXPONENT'] = simulation.config.noise_contributions.optical_path_difference_variability.power_law_exponent
        header['LIFESIM2_OPD_VARIABILITY_RMS'] = str(
            simulation.config.noise_contributions.optical_path_difference_variability.rms)
        header['LIFESIM2_ADJUST_BASELINE_TO_HABITABLE_ZONE'] = simulation.observation.adjust_baseline_to_habitable_zone
        header['LIFESIM2_INTEGRATION_TIME'] = str(simulation.observation.integration_time)
        header['LIFESIM2_OPTIMIZED_WAVELENGTH'] = str(simulation.observation.optimized_wavelength)
        header['LIFESIM2_ARRAY_CONFIGURATION_TYPE'] = simulation.observation.observatory.array_configuration.type
        header['LIFESIM2_BASELINE_MAXIMUM'] = str(
            simulation.observation.observatory.array_configuration.baseline_maximum)
        header['LIFESIM2_BASELINE_MINIMUM'] = str(
            simulation.observation.observatory.array_configuration.baseline_minimum)
        header['LIFESIM2_BASELINE_RATIO'] = simulation.observation.observatory.array_configuration.baseline_ratio
        header['LIFESIM2_MODULATION_PERIOD'] = str(
            simulation.observation.observatory.array_configuration.modulation_period)
        header[
            'LIFESIM2_BEAM_COMBINATION_SCHEME'] = simulation.observation.observatory.beam_combination_scheme.type.value
        header['LIFESIM2_APERTURE_DIAMETER'] = str(
            simulation.observation.observatory.instrument_parameters.aperture_diameter)
        header[
            'LIFESIM2_SPECTRAL_RESOLVING_POWER'] = simulation.observation.observatory.instrument_parameters.spectral_resolving_power
        header['LIFESIM2_WAVELENGTH_RANGE_LOWER_LIMIT'] = str(
            simulation.observation.observatory.instrument_parameters.wavelength_range_lower_limit)
        header['LIFESIM2_WAVELENGTH_RANGE_UPPER_LIMIT'] = str(
            simulation.observation.observatory.instrument_parameters.wavelength_range_upper_limit)
        header[
            'LIFESIM2_UNPERTURBED_INSTRUMENT_THROUGHPUT'] = simulation.observation.observatory.instrument_parameters.unperturbed_instrument_throughput
        for index_source, source in enumerate(simulation.observation.sources.values()):
            if isinstance(source, Planet):
                header[f'LIFESIM2_PLANET{index_source}_NAME'] = source.name
                header[f'LIFESIM2_PLANET{index_source}_MASS'] = str(source.mass)
                header[f'LIFESIM2_PLANET{index_source}_RADIUS'] = str(source.radius)
                header[f'LIFESIM2_PLANET{index_source}_STAR_SEPARATION_X'] = str(source.star_separation_x)
                header[f'LIFESIM2_PLANET{index_source}_STAR_SEPARATION_Y'] = str(source.star_separation_y)
                header[f'LIFESIM2_PLANET{index_source}_TEMPERATURE'] = str(source.temperature)
            if isinstance(source, Star):
                header['LIFESIM2_STAR_NAME'] = source.name
                header['LIFESIM2_STAR_DISTANCE'] = str(source.distance)
                header['LIFESIM2_STAR_MASS'] = str(source.mass)
                header['LIFESIM2_STAR_RADIUS'] = str(source.radius)
                header['LIFESIM2_STAR_TEMPERATURE'] = str(source.temperature)
                header['LIFESIM2_STAR_LUMINOSITY'] = str(source.luminosity)
                header['LIFESIM2_STAR_ZODI_LEVEL'] = source.zodi_level
        return header

    @staticmethod
    def write_fits(output_path: Path, postfix: str, simulation: Simulation, differential_photon_counts):
        """Write the differential photon counts to a FITS file.

        :param output_path: The output path of the FITS file
        :param postfix: Postfix that is appended to the output file name
        :param simulation: The simulation object
        :param differential_photon_counts: The differential photon counts
        """
        hdu_list = []
        primary = fits.PrimaryHDU()
        header = FITSWriter._get_fits_header(primary, simulation)
        hdu_list.append(primary)
        for index_response in differential_photon_counts.keys():
            differential_photon_counts_list = list(differential_photon_counts[index_response].values())
            differential_photon_counts_array = np.array(differential_photon_counts_list)
            hdu = fits.ImageHDU(differential_photon_counts_array)
            hdu_list.append(hdu)
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(f'differential_photon_counts_{datetime.now().strftime("%Y%m%d_%H%M%S")}{postfix}.fits')
