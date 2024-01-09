import typing

import numpy as np


def _matrix_to_bits(matrix: np.ndarray):
    return np.unpackbits(matrix.view(np.uint8), bitorder="big").reshape((matrix.shape[0], -1))


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


def deapply_g17_transformation(a: np.ndarray, shape_0: int, dtype_size: int):
    x = a.flatten()
    y = np.unpackbits(x, bitorder="big").reshape((-1, 8))
    y = y.reshape((-1, shape_0))
    tt = y.transpose()
    f = np.packbits(tt, bitorder="big").view(dtype=np.dtype(f"c{dtype_size}"))
    return f.reshape((shape_0, -1))


if __name__ == '__main__':
    x = np.array([
        [127 + 1j, 2, 8],
        [3, 4, 9]
    ], dtype=np.complex128)
    s = x.shape
    ger = apply_g17_transformation(x)
    print("\n")
    y = deapply_g17_transformation(ger, s[0], x.dtype.itemsize)
    print(x, y)
    assert (x == y).all()


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
