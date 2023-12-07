from pathlib import Path
from typing import Tuple

from sygn.core.context import Context
from sygn.core.modules.base_module import BaseModule
from sygn.io.fits_writer import FITSWriter
from sygn.util.helpers import FITSDataType


class FITSWriterModule(BaseModule):
    """Class representation of the FITS writer module.
    """

    def __init__(self, output_path: Path, data_type: Tuple[FITSDataType]):
        """Constructor method.

        :param output_path: Output path of the FITS file
        :param data_type: Type of the data to be written to a FITS file
        """
        self._output_path = output_path
        self._data_type = data_type
        self.dependencies = []

    def apply(self, context: Context) -> Context:
        """Write the FITS file.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        FITSWriter.write_fits(self._output_path, context, self._data_type)
        return context
