class Template():
    """Class representation of a signal template.
    """

    def __init__(self, signal, effective_area_rms, index_x, index_y):
        """Constructor method.
        """
        self.signal = signal
        self.effective_area_rms = effective_area_rms
        self.index_x = None
        self.index_y = None
