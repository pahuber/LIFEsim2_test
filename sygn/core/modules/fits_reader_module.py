from pathlib import Path
from typing import Tuple

from sygn.core.context import Context
from sygn.io.fits_reader import FITSReader
from sygn.util.helpers import FITSDataType


class FITSReaderModule():
    """Class representation of the FITS reader module.
    """

    def __init__(self, input_path: Path, data_type: Tuple[FITSDataType]):
        """Constructor method.

        :param input_path: Input path of the FITS file
        :param data_type: Type of the data to be written to a FITS file
        """
        self._input_path = input_path
        self._data_type = data_type

    def apply(self, context: Context) -> Context:
        """Read the FITS file and load the settings, mission, observatory and photon sources.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        context = FITSReader.read_fits(self._input_path, context, self._data_type)
        return context
