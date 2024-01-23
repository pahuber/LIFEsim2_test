class Extraction():
    def __init__(self, spectrum, spectrum_uncertainties, cost_function):
        """Constructor method.
        """
        self.spectrum = spectrum
        self.spectrum_uncertainties = spectrum_uncertainties
        self.cost_function = cost_function
        self.spectrum_fit = None
