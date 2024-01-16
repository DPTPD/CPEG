import typing

import numpy as np


def _matrix_to_bits(matrix: np.ndarray):
    return np.unpackbits(np.ascontiguousarray(matrix).view(np.uint8), bitorder="big").reshape((matrix.shape[0], -1))


def _apply_g17_transformation_to_bits(arr: np.ndarray, sh):
    arr = np.transpose(arr)
    temp = arr.reshape((-1, 8))
    temp2 = np.packbits(temp, bitorder="big")
    x = temp2.reshape((sh[0], -1))
    return x


def apply_g17_transformation(a: np.ndarray):
    s = a.shape
    test = _matrix_to_bits(a)
    return _apply_g17_transformation_to_bits(test, s)


def deapply_g17_transformation(a: np.ndarray, dtype_size: int):
    shape_0 = a.shape[0]
    x = a.flatten()
    y = np.unpackbits(x, bitorder="big").reshape((-1, shape_0))
    tt = y.transpose()
    f = np.packbits(tt, bitorder="big").view(dtype=np.dtype(f"f{dtype_size}"))
    return f.reshape((shape_0, -1))


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

    def deapply(self, mat: np.ndarray, dtype_size: int) -> np.ndarray:
        if self.g17 is None:
            return mat
        if self.g17 == "bits":
            return deapply_g17_transformation(mat, dtype_size)
        if self.g17 == "bytes":
            return mat.T
        raise NotImplementedError
