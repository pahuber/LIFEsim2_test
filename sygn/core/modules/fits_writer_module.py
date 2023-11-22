from pathlib import Path

from sygn.core.context import Context
from sygn.io.fits_writer import FITSWriter


class FITSWriterModule():
    """Class representation of the FITS writer module.
    """

    def __init__(self, output_path: Path):
        """Constructor method.

        :param output_path: Output path of the FITS file
        """
        self._output_path = output_path

    def apply(self, context: Context) -> Context:
        """Load the configurations from the config file and initialize the settings, mission and observatory objects.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        FITSWriter.write_fits(self._output_path, context)
        return context
