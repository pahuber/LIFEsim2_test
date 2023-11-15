from sygn.core.pipeline.base_pipeline import BasePipeline
from sygn.core.context.processor_context import ProcessorContext


class ProcessorPipeline(BasePipeline):
    """Class representation of the processor pipeline.
    """

    def __init__(self):
        """Constructor method.
        """
        super().__init__()
        self.context = ProcessorContext()

    def _validate_modules(self):
        """Validate that the correct number of each module type is added to the pipeline.
        """
        pass

    def get_data(self) -> list:
        """Get the synthetic data.

        :return: The data
        """
        pass
