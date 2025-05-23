"""Parent class for analysis of single processes in the network."""
from .network_analysis import NetworkAnalysis


class SingleProcessAnalysis(NetworkAnalysis):
    def __init__(self):
        self._cmi_estimator = None
        super().__init__()
