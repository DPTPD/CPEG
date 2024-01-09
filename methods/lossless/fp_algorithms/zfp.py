import typing

import numpy as np
import zfpy

from methods.general.floatifier import Floatifier
from methods.general.g17_transform import G17Transformer
from methods.lossless.fp_algorithms.fp_algorithm import FpAlgorithm


class ZfpCompressor(FpAlgorithm):
    def is_lossless(self) -> bool:
        return self.precision == -1

    def __init__(self, floatifier: Floatifier | typing.Literal["in_place", "hstack", "vstack"],
                 g17: G17Transformer | typing.Literal[None, "bytes"], precision: int = -1):
        super().__init__(floatifier, g17, precision)

    def _compress_float_matrix(self, matrix: np.ndarray) -> bytes:
        return zfpy.compress_numpy(matrix, precision=self.precision)

    def _decompress_float_matrix(self, data: bytes) -> np.ndarray:
        return zfpy.decompress_numpy(data)
