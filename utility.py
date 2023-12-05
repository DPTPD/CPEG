import zfpy
import numpy as np


def compress_fpzip(matrix: np.array, precision=0):
    assert matrix.dtype
    compressed_data_real = zfpy.compress_numpy(matrix.real, precision=precision)
    compressed_data_imag = zfpy.compress_numpy(matrix.imag, precision=precision)
    return compressed_data_real, compressed_data_imag


def decompress_fpzip(compressed_real, compressed_imag):
    decompressed_array_real = zfpy.decompress_numpy(compressed_real)
    decompressed_array_imag = zfpy.decompress_numpy(compressed_imag)
    return decompressed_array_real + (decompressed_array_imag * 1j)

