from lifesim2.observatory.beam_combination_schemes import DoubleBracewell4
from lifesim2.observatory.instrument_specification import InstrumentSpecification
from lifesim2.observatory.observatory import Observatory


observatory = Observatory()
observatory.load_from_config(r'C:\Users\huber\Desktop\LIFEsim2\observatory.yaml')

print(observatory.array_configuration)
print(observatory.beam_combination_scheme)