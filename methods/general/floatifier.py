import typing

import numpy as np


class Floatifier:

    def __init__(self, floatifier: typing.Literal["in_place", "hstack", "vstack"]):
        assert floatifier in ["in_place", "hstack", "vstack"]
        self.floatifier = floatifier

    def apply(self, mat: np.ndarray) -> np.ndarray:
        if self.floatifier == "in_place":
            return np.ascontiguousarray(mat).view(np.dtype("f" + str(mat.dtype.itemsize // 2)))
        if self.floatifier == "hstack":
            return np.hstack((np.real(mat), np.imag(mat)))
        if self.floatifier == "vstack":
            return np.vstack((np.real(mat), np.imag(mat)))
        raise NotImplementedError

    def deapply(self, mat: np.ndarray) -> np.ndarray:
        if self.floatifier == "in_place":
            return np.ascontiguousarray(mat).view(np.dtype("c" + str(mat.dtype.itemsize * 2)))
        if self.floatifier == "hstack":
            split_point = mat.shape[1] // 2
            real = mat[:, :split_point]
            imag = mat[:, split_point:]
            return real + 1j * imag
        if self.floatifier == "vstack":
            split_point = mat.shape[0] // 2
            real = mat[:split_point, :]
            imag = mat[split_point:, :]
            return real + 1j * imag
        raise NotImplementedError
