from sygn.core.contexts.processor_context import ProcessorContext
from sygn.core.pipelines.base_pipeline import BasePipeline


class ProcessorPipeline(BasePipeline):
    """Class representation of the processor pipelines.
    """

    def __init__(self):
        """Constructor method.
        """
        super().__init__()
        self.context = ProcessorContext()

    def _validate_modules(self):
        """Validate that the correct number of each modules type is added to the pipelines.
        """
        pass

    def get_data(self) -> list:
        """Get the synthetic data.

        :return: The data
        """
        pass

    def run(self):
        """Run the pipelines by calling the apply method of each modules.
        """
        self._validate_modules()
        for module in self._modules:
            context = module.apply(context=self.context)
