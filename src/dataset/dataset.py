import numpy as np
import numpy.typing as npt


class Dataset:
    def __init__(self, X: npt.NDArray[np.float32], y: npt.NDArray[np.int_]) -> None:
        if X.ndim != 2: raise ValueError("X matrix must be 2d array")
        if y.ndim != 1: raise ValueError("y matrix must be 1d array")
        if X.shape[0] != y.shape[0]: raise ValueError("X and y matrix must have same number of samples")

        self.X = X
        self.y = y


    def __len__(self) -> int:
        return len(self.X)