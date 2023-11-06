from sygn.core.context import Context
from sygn.core.modules.base_module import BaseModule
from sygn.core.modules.data_generator.data_generator import DataGenerator, DataGenerationMode


class DataGeneratorModule(BaseModule):
    def __init__(self, mode: DataGenerationMode):
        self.mode = mode
        self.data_generator = None

    def apply(self, context: Context) -> Context:
        time_range = context.time_range
        self.data_generator = DataGenerator(mode=self.mode,
                                            settings=context.settings,
                                            observation=context.observation,
                                            observatory=context.observatory,
                                            target_systems=context.target_systems,
                                            time_range=time_range,
                                            animator=None)
        # animator=animator)
        self.data_generator.run()
        context.differential_photon_counts = self.data_generator.output.differential_photon_counts
        return context
