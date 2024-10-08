import typing

import fpzip
import numpy as np

from methods.general.floatifier import Floatifier
from methods.general.g17_transform import G17Transformer
from methods.lossless.fp_algorithms.fp_algorithm import FpAlgorithm


class FpzipCompressor(FpAlgorithm):
    def is_lossless(self) -> bool:
        return self.precision == 0

    def __init__(self, floatifier: Floatifier | typing.Literal["in_place", "hstack", "vstack"],
                 g17: G17Transformer | typing.Literal[None, "bytes"], precision: int = 0):
        super().__init__(floatifier, g17, precision)

    def _compress_float_matrix(self, matrix: np.ndarray) -> bytes:
        return fpzip.compress(matrix, precision=self.precision)

    def _decompress_float_matrix(self, data: bytes) -> np.ndarray:
        matrix = fpzip.decompress(data)
        # decomprimere fpzip restituisce sempre un tensore a 4 dimensioni
        matrix = matrix.reshape((-1,))
        return matrix
