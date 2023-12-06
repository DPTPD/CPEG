import math
import zlib

import numpy as np

from methods.general import compressor


def _complex_to_bits(matrix: np.ndarray):
    return np.unpackbits(np.array([matrix]).view(np.uint8)).view(np.bool_)


def _matrix_to_bits(matrix: np.ndarray):
    sh = matrix.shape
    arr2 = np.zeros((math.prod(sh), matrix.dtype.itemsize * 8), dtype=np.bool_)
    for idx, x in enumerate(np.nditer(matrix)):
        converted = _complex_to_bits(x)
        arr2[idx, :] = converted

    return arr2


def _apply_ger_transformation_to_bits(arr: np.ndarray):
    arr = np.transpose(arr)
    temp = arr.reshape((-1, 8))
    temp2 = np.packbits(temp)
    return temp2


def apply_ger_transformation(a: np.ndarray):
    test = _matrix_to_bits(a)
    return _apply_ger_transformation_to_bits(test)


class GerCompressor(compressor.Compressor):

    def compress(self, hologram: compressor.HoloSpec, output_path: str):
        compressed_data = apply_ger_transformation(hologram.holo)
        data = zlib.compress(compressed_data.tobytes(), level=9)
        with open(output_path, "wb") as fp:
            fp.write(data)

    def decompress(self, input_path: str) -> compressor.HoloSpec:
        raise NotImplementedError
