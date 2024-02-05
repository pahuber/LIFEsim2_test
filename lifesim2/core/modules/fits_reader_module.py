import glob
from pathlib import Path
from typing import Tuple

import numpy as np
from astropy import units as u

from lifesim2.core.context import Context
from lifesim2.core.modules.base_module import BaseModule
from lifesim2.core.modules.config_loader_module import ConfigLoaderModule
from lifesim2.core.modules.target_loader_module import TargetLoaderModule
from lifesim2.core.template import Template
from lifesim2.io.fits_reader import FITSReader
from lifesim2.util.helpers import FITSReadWriteType


class FITSReaderModule(BaseModule):
    """Class representation of the FITS reader module.
    """

    def __init__(self, input_path: Path, data_type: Tuple[FITSReadWriteType]):
        """Constructor method.

        :param input_path: Input path of the FITS file
        :param data_type: Type of the data to be written to a FITS file
        """
        self._input_path = input_path
        self._data_type = data_type
        self.dependencies = []

    def _create_entities_from_fits_header(self, context, data_fits_header) -> Context:
        """Create dictionaries from the FITS header and load the entities from the dictionaries.

        :param context: The context object
        :param data_fits_header: The FITS header of the data
        :return: The context object
        """
        config_dict = FITSReader._create_config_dict_from_fits_header(data_fits_header)
        target_dict = FITSReader._create_target_dict_from_fits_header(data_fits_header)
        context = ConfigLoaderModule(path_to_config_file=None, config_dict=config_dict).apply(context)
        context = TargetLoaderModule(path_to_context_file=None, config_dict=target_dict).apply(context)
        return context

    def apply(self, context: Context) -> Context:
        """Read the FITS file(s) and load the (template) signals. If the file corresponds to a synthetic measurement,
        load the settings, mission, observatory and photon sources.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        if self._data_type == FITSReadWriteType.SyntheticMeasurement:
            context.signal, data_fits_header, _ = FITSReader.read_fits(
                self._input_path,
                context)
            context = self._create_entities_from_fits_header(context, data_fits_header)

        elif self._data_type == FITSReadWriteType.Template:
            context.templates = np.zeros((context.settings.grid_size, context.settings.grid_size), dtype=object)
            fits_files = glob.glob(f"{self._input_path}/*.fits")

            for fits_file in fits_files:
                template_signal, template_fits_header, effective_area_rms = FITSReader.read_fits(fits_file, context)

                # Check that template properties match data properties
                FITSReader._check_template_fits_header(context, template_fits_header)

                index_x, index_y = FITSReader._read_indices_from_fits_header(template_fits_header)

                context.templates[index_x, index_y] = Template(template_signal, effective_area_rms * u.m ** 2, index_x,
                                                               index_y)

        return context
