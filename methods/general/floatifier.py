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
            return np.hstack((mat.real, mat.imag))
        if self.floatifier == "vstack":
            return np.vstack((mat.real, mat.imag))
        raise NotImplementedError

    def deapply(self, mat: np.ndarray) -> np.ndarray:
        if self.floatifier == "in_place":
            return np.ascontiguousarray(mat).view(np.dtype("c" + str(mat.dtype.itemsize * 2)))
        if self.floatifier == "hstack":
            real, imag = np.hsplit(mat, 2)
            return real + 1j * imag
        if self.floatifier == "vstack":
            real, imag = np.vsplit(mat, 2)
            return real + 1j * imag
        raise NotImplementedError
