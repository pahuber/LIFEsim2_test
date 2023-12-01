import glob
from pathlib import Path

import numpy as np
from astropy.io import fits

from sygn.core.context import Context
from sygn.util.helpers import FITSDataType


class FITSReader():
    """Class representation of the FITS reader.
    """

    @staticmethod
    def _extract_data(data: np.ndarray) -> np.ndarray:
        """Extract and return the data arrays from the FITS file.

        :param data: The FITS data
        :return: The data arrays
        """
        data_array = np.zeros(((len(data),) + data[0].shape))
        for index_data in range(len(data)):
            data_array[index_data] = data[index_data].data
        return data_array

    @staticmethod
    def read_fits(input_path: Path, context: Context, data_type: FITSDataType) -> Context:
        """Read the settings, mission, observatory and photon sources from the FITS file.

        :param input_path: The input path of the FITS file
        :param context: The context
        :param data_type: The data type to be written to FITS
        :return: The context containing the data or templates
        """
        if data_type == FITSDataType.SyntheticMeasurement:
            with fits.open(input_path) as hdul:
                context.data = FITSReader._extract_data(data=hdul[1:])
        elif data_type == FITSDataType.Template:
            fits_files = glob.glob(f"{input_path}/*.fits")
            for fits_file in fits_files:
                with fits.open(fits_file) as hdul:
                    context.templates.append(FITSReader._extract_data(data=hdul[1:]))
        return context
