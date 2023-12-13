from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits

from sygn.core.context import Context


class FITSReader():
    """Class representation of the FITS reader.
    """

    @staticmethod
    def _check_template_fits_header(context: Context, template_fits_header):
        if context.settings.grid_size != template_fits_header['SYGN_GRID_SIZE']:
            raise ValueError(f'Template grid size does not match data grid size')
        # TODO: check other properties

    @staticmethod
    def _create_config_dict_from_fits_header(data_fits_header):
        config_dict = {}
        config_dict['settings'] = {
            'grid_size': data_fits_header['SYGN_GRID_SIZE'],
            'time_step': data_fits_header['SYGN_TIME_STEP'],
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
            'adjust_baseline_to_habitable_zone': data_fits_header['SYGN_ADJUST_BASELINE_TO_HABITABLE_ZONE'],
            'integration_time': data_fits_header['SYGN_INTEGRATION_TIME'],
            'optimized_wavelength': data_fits_header['SYGN_OPTIMIZED_WAVELENGTH']
        }
        config_dict['observatory'] = {
            'array_configuration': {
                'type': data_fits_header['SYGN_ARRAY_CONFIGURATION_TYPE'],
                'baseline_maximum': data_fits_header['SYGN_BASELINE_MAXIMUM'],
                'baseline_minimum': data_fits_header['SYGN_BASELINE_MINIMUM'],
                'baseline_ratio': data_fits_header['SYGN_BASELINE_RATIO'],
                'modulation_period': data_fits_header['SYGN_MODULATION_PERIOD']
            },
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
    def _create_target_dict_from_fits_header(data_fits_header):
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
    def _extract_effective_area_from_fits_header(template_fits_header, context):
        effective_area = np.zeros(context.observatory.beam_combination_scheme.number_of_differential_outputs) * u.m ** 2
        for index_output in range(context.observatory.beam_combination_scheme.number_of_differential_outputs):
            effective_area[index_output] = u.Quantity(template_fits_header[f'SYGN_EFFECTIVE_AREA_{index_output}'])
        return effective_area

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
