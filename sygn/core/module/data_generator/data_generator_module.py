from sygn.core.context.base_context import BaseContext
from sygn.core.module.base_module import BaseModule
from sygn.core.module.data_generator.data_generator import DataGenerator, DataGenerationMode


class DataGeneratorModule(BaseModule):
    """Class representation of the data generator module.
    """

    def __init__(self, mode: DataGenerationMode):
        """Constructor method.

        :param mode: Data generation mode
        """
        self.mode = mode
        self.data_generator = None

    def apply(self, context: BaseContext) -> BaseContext:
        """Apply the module.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        time_range = context.time_range
        self.data_generator = DataGenerator(mode=self.mode,
                                            settings=context.settings,
                                            observation=context.observation,
                                            observatory=context.observatory,
                                            target_systems=context.target_systems,
                                            time_range=time_range,
                                            animator=context.animator)
        self.data_generator.run()
        for index_target_system in range(len(self.data_generator.output)):
            context.differential_photon_counts_list.append(
                self.data_generator.output[index_target_system].differential_photon_counts)
        return context
