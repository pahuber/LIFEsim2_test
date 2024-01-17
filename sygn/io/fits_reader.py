from pathlib import Path
from typing import Tuple

import numpy as np
from astropy import units as u
from astropy.io import fits

from sygn.core.context import Context


class FITSReader():
    """Class representation of the FITS reader.
    """

    @staticmethod
    def _check_template_fits_header(context: Context, template_fits_header: fits.header.Header):
        """Check that the relevant template configuration and the data configuration match.

        :param context: The context object
        :param template_fits_header: The template FITS header
        """
        if context.settings.grid_size != template_fits_header['SYGN_GRID_SIZE']:
            raise ValueError(f'Template grid size does not match data grid size')
        if context.settings.time_steps != int(template_fits_header['SYGN_TIME_STEPS']):
            raise ValueError(f'Template time steps do not match data time steps')
        if context.settings.planet_orbital_motion != template_fits_header['SYGN_PLANET_ORBITAL_MOTION']:
            raise ValueError(f'Template planet orbital motion flag does not match data planet orbital motion')
        if context.mission.integration_time.to(u.d) / context.mission.modulation_period.to(u.d) != \
                u.Quantity(template_fits_header['SYGN_INTEGRATION_TIME']).to(u.d) / u.Quantity(
            template_fits_header['SYGN_MODULATION_PERIOD']).to(u.d):
            raise ValueError(
                f'Ratio of integration time to modulation period of data does not match the template ratio')
        if context.mission.baseline_ratio != template_fits_header['SYGN_BASELINE_RATIO']:
            raise ValueError(f'Template baseline ratio does not match data baseline ratio')
        if context.mission.baseline_maximum != template_fits_header['SYGN_BASELINE_MAXIMUM']:
            raise ValueError(f'Template baseline maximum does not match data baseline maximum')
        if context.mission.baseline_minimum != template_fits_header['SYGN_BASELINE_MINIMUM']:
            raise ValueError(f'Template baseline minimum does not match data baseline minimum')
        if context.mission.optimized_differential_output != int(
                template_fits_header['SYGN_OPTIMIZED_DIFFERENTIAL_OUTPUT']):
            raise ValueError(f'Template optimized differential output flag does not match data optimized differential '
                             f'output')
        if context.mission.optimized_star_separation != template_fits_header['SYGN_OPTIMIZED_STAR_SEPARATION']:
            raise ValueError(f'Template optimized star separation does not match data optimized star separation')
        if context.mission.optimized_wavelength != template_fits_header['SYGN_OPTIMIZED_WAVELENGTH']:
            raise ValueError(f'Template optimized wavelength does not match data optimized wavelength')
        if context.observatory.array_configuration.type.value != template_fits_header['SYGN_ARRAY_CONFIGURATION_TYPE']:
            raise ValueError(f'Template array configuration does not match data array configuration')
        if context.observatory.beam_combination_scheme.type != template_fits_header['SYGN_BEAM_COMBINATION_SCHEME']:
            raise ValueError(f'Template beam combination scheme does not match data beam combination scheme')
        if context.observatory.instrument_parameters.spectral_resolving_power != \
                int(template_fits_header['SYGN_SPECTRAL_RESOLVING_POWER']):
            raise ValueError(f'Template spectral resolving power does not match data spectral resolving power')
        if context.observatory.instrument_parameters.wavelength_range_lower_limit != \
                template_fits_header['SYGN_WAVELENGTH_RANGE_LOWER_LIMIT']:
            raise ValueError(f'Template wavelength range lower limit does not match data wavelength range lower limit')
        if context.observatory.instrument_parameters.wavelength_range_upper_limit != \
                template_fits_header['SYGN_WAVELENGTH_RANGE_UPPER_LIMIT']:
            raise ValueError(f'Template wavelength range upper limit does not match data wavelength range upper limit')
        if context.star.distance != template_fits_header['SYGN_STAR_DISTANCE']:
            raise ValueError(f'Template star distance does not match data star distance.')
        if context.mission.optimized_star_separation == 'habitable-zone' and context.star.temperature != \
                template_fits_header['SYGN_STAR_TEMPERATURE']:
            raise ValueError(
                f'Template star temperature does not match data star temperature. This is an issue since the habitable zone is dependent on this quantity.')
        if context.mission.optimized_star_separation == 'habitable-zone' and context.star.luminosity != \
                template_fits_header['SYGN_STAR_LUMINOSITY']:
            raise ValueError(
                f'Template star luminosity does not match data star luminosity. This is an issue since the habitable zone is dependent on this quantity.')

    @staticmethod
    def _create_config_dict_from_fits_header(data_fits_header: fits.header.Header) -> dict:
        """Create the configuration dictionary from the FITS header.

        :param data_fits_header: The FITS header
        :return: The configuration dictionary
        """
        config_dict = {}
        config_dict['settings'] = {
            'grid_size': data_fits_header['SYGN_GRID_SIZE'],
            'time_steps': data_fits_header['SYGN_TIME_STEPS'],
            'planet_orbital_motion': data_fits_header['SYGN_PLANET_ORBITAL_MOTION'],
            'noise_contributions': {
                'stellar_leakage': data_fits_header['SYGN_STELLAR_LEAKAGE'],
                'local_zodi_leakage': data_fits_header['SYGN_LOCAL_ZODI_LEAKAGE'],
                'exozodi_leakage': data_fits_header['SYGN_EXOZODI_LEAKAGE'],
                'fiber_injection_variability': data_fits_header['SYGN_FIBER_INJECTION_VARIABILITY'],
                'optical_path_difference_variability': {
                    'apply': data_fits_header['SYGN_OPD_VARIABILITY_APPLY '],
                    'power_law_exponent': data_fits_header['SYGN_OPD_VARIABILITY_POWER_LAW_EXPONENT'],
                    'rms': data_fits_header['SYGN_OPD_VARIABILITY_RMS']
                }
            }
        }
        config_dict['mission'] = {
            'integration_time': data_fits_header['SYGN_INTEGRATION_TIME'],
            'modulation_period': data_fits_header['SYGN_MODULATION_PERIOD'],
            'baseline_ratio': data_fits_header['SYGN_BASELINE_RATIO'],
            'baseline_maximum': data_fits_header['SYGN_BASELINE_MAXIMUM'],
            'baseline_minimum': data_fits_header['SYGN_BASELINE_MINIMUM'],
            'optimized_differential_output': data_fits_header['SYGN_OPTIMIZED_DIFFERENTIAL_OUTPUT'],
            'optimized_star_separation': data_fits_header['SYGN_OPTIMIZED_STAR_SEPARATION'],
            'optimized_wavelength': data_fits_header['SYGN_OPTIMIZED_WAVELENGTH']
        }
        config_dict['observatory'] = {
            'array_configuration': data_fits_header['SYGN_ARRAY_CONFIGURATION_TYPE'],
            'beam_combination_scheme': data_fits_header['SYGN_BEAM_COMBINATION_SCHEME'],
            'instrument_parameters': {
                'aperture_diameter': data_fits_header['SYGN_APERTURE_DIAMETER'],
                'spectral_resolving_power': data_fits_header['SYGN_SPECTRAL_RESOLVING_POWER'],
                'wavelength_range_lower_limit': data_fits_header['SYGN_WAVELENGTH_RANGE_LOWER_LIMIT'],
                'wavelength_range_upper_limit': data_fits_header['SYGN_WAVELENGTH_RANGE_UPPER_LIMIT'],
                'unperturbed_instrument_throughput': data_fits_header['SYGN_UNPERTURBED_INSTRUMENT_THROUGHPUT']
            }
        }
        return config_dict

    @staticmethod
    def _create_target_dict_from_fits_header(data_fits_header: fits.header.Header) -> dict:
        """Create the target dictionary from the FITS header.

        :param data_fits_header: The FITS header
        :return: The target dictionary
        """
        target_dict = {}
        target_dict['star'] = {
            'name': data_fits_header['SYGN_STAR_NAME'],
            'distance': data_fits_header['SYGN_STAR_DISTANCE'],
            'mass': data_fits_header['SYGN_STAR_MASS'],
            'radius': data_fits_header['SYGN_STAR_RADIUS'],
            'temperature': data_fits_header['SYGN_STAR_TEMPERATURE'],
            'luminosity': data_fits_header['SYGN_STAR_LUMINOSITY'],
            'right_ascension': data_fits_header['SYGN_STAR_RIGHT_ASCENSION'],
            'declination': data_fits_header['SYGN_STAR_DECLINATION']
        }
        try:
            target_dict['zodi'] = {
                'level': data_fits_header['SYGN_EXOZODI_LEVEL'],
                'inclination': data_fits_header['SYGN_EXOZODI_INCLINATION']
            }
        except KeyError:
            pass
        target_dict['planets'] = {}
        planet_counter = 1
        for key in data_fits_header:
            if key.startswith('SYGN_PLANET_') and not key == 'SYGN_PLANET_ORBITAL_MOTION':
                planet_name = key.split('_')[2]
                planet_key = f'planet_{planet_counter}'
                if planet_key not in target_dict['planets']:
                    target_dict['planets'][planet_key] = {}
                    target_dict['planets'][planet_key]['name'] = planet_name
                if planet_name.upper() in key.upper():
                    target_dict['planets'][planet_key][key.replace(f'SYGN_PLANET_{planet_name}_', '').lower()] = \
                        data_fits_header[key]
        return target_dict

    @staticmethod
    def _read_indices_from_fits_header(template_fits_header: fits.header.Header) -> Tuple:
        """Read the indices from the FITS header.

        :param template_fits_header: The FITS header
        :return: A tuple containing the indices
        """
        index_x = template_fits_header['SYGN_INDEX_X']
        index_y = template_fits_header['SYGN_INDEX_Y']
        return index_x, index_y

    @staticmethod
    def _extract_data(data: np.ndarray) -> np.ndarray:
        """Extract and return the data arrays from the FITS file.

        :param data: The FITS data
        :return: The data arrays
        """
        images = [image for image in data if type(image) == fits.hdu.image.ImageHDU]

        data_array = np.zeros(((len(images),) + data[0].shape))
        effective_area_array = np.zeros(data[0].shape[0])
        for index_data in range(len(data)):
            if type(data[index_data]) == fits.hdu.image.ImageHDU:
                data_array[index_data] = data[index_data].data
            elif type(data[index_data]) == fits.hdu.table.BinTableHDU:
                effective_area_array = data[index_data].data.field(0)
        return data_array, effective_area_array

    @staticmethod
    def read_fits(input_path: Path, context: Context) -> tuple:
        """Read the photon count data from the FITS file.

        :param input_path: The input path of the FITS file
        :param context: The context
        :return: The context containing the data or templates
        """
        with fits.open(input_path) as hdul:
            header = hdul[0].header
            data, effective_area = FITSReader._extract_data(data=hdul[1:])
        return data, header, effective_area
