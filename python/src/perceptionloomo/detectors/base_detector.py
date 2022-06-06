import numpy as np

class BaseDetector():
    def __init__(self, verbose) -> None:
        self.verbose = verbose
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Detector Base Class does not provide a predict class.")