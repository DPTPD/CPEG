import math
import typing

import numpy as np


def _complex_to_bits(matrix: np.ndarray):
    return np.unpackbits(np.array([matrix]).view(np.uint8)).view(np.bool_)


def _matrix_to_bits(matrix: np.ndarray):
    sh = matrix.shape
    arr2 = np.zeros((math.prod(sh), matrix.dtype.itemsize * 8), dtype=np.bool_)
    for idx, x in enumerate(np.nditer(matrix)):
        converted = _complex_to_bits(x)
        arr2[idx, :] = converted

    return arr2


def _apply_g17_transformation_to_bits(arr: np.ndarray):
    arr = np.transpose(arr)
    temp = arr.reshape((-1, 8))
    temp2 = np.packbits(temp)
    return temp2


def apply_g17_transformation(a: np.ndarray):
    test = _matrix_to_bits(a)
    return _apply_g17_transformation_to_bits(test)


class G17Transformer:
    def __init__(self, g17: typing.Literal[None, "bits", "bytes"]):
        assert g17 in [None, "bits", "bytes"]
        self.g17 = g17

    def apply(self, mat: np.ndarray) -> np.ndarray:
        if self.g17 is None:
            return mat
        if self.g17 == "bits":
            return apply_g17_transformation(mat)
        if self.g17 == "bytes":
            return mat.T
        raise NotImplementedError

    def deapply(self, mat: np.ndarray) -> np.ndarray:
        if self.g17 is None:
            return mat
        if self.g17 == "bits":
            raise NotImplementedError
        if self.g17 == "bytes":
            return mat.T
        raise NotImplementedError
