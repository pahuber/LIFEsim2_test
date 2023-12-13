import glob
from pathlib import Path
from typing import Tuple

from sygn.core.context import Context
from sygn.core.modules.base_module import BaseModule
from sygn.core.modules.config_loader_module import ConfigLoaderModule
from sygn.core.modules.target_loader_module import TargetLoaderModule
from sygn.io.fits_reader import FITSReader
from sygn.util.helpers import FITSDataType


class FITSReaderModule(BaseModule):
    """Class representation of the FITS reader module.
    """

    def __init__(self, input_path: Path, data_type: Tuple[FITSDataType]):
        """Constructor method.

        :param input_path: Input path of the FITS file
        :param data_type: Type of the data to be written to a FITS file
        """
        self._input_path = input_path
        self._data_type = data_type
        self.dependencies = []

    def _create_entities_from_fits_header(self, context, data_fits_header) -> Context:
        config_dict = FITSReader._create_config_dict_from_fits_header(data_fits_header)
        context = ConfigLoaderModule(path_to_config_file=None, config_dict=config_dict).apply(context)
        target_dict = FITSReader._create_target_dict_from_fits_header(data_fits_header)
        context = TargetLoaderModule(path_to_context_file=None, config_dict=target_dict).apply(context)
        return context

    def apply(self, context: Context) -> Context:
        """Read the FITS file and load the settings, mission, observatory and photon sources.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        if self._data_type == FITSDataType.SyntheticMeasurement:
            context.data, data_fits_header, _ = FITSReader.read_fits(self._input_path, context)

            # Create the configuration entities from the FITS header, if it is not provided through the
            # ConfigLoaderModule
            context = self._create_entities_from_fits_header(context, data_fits_header)

            # TODO: Create the photon source entities from the FITS header, if they are needed, i.e. if there is a
            # TemplateGeneratorModule and they are not provided through the TargetLoaderModule

        elif self._data_type == FITSDataType.Template:
            fits_files = glob.glob(f"{self._input_path}/*.fits")
            for fits_file in fits_files:
                template, template_fits_header, effective_area = FITSReader.read_fits(fits_file, context)

                # Check that template properties match data properties
                FITSReader._check_template_fits_header(context, template_fits_header)

                # Extract effective areas
                context.effective_area_rms.append(effective_area)

                context.templates.append(template)
        return context
