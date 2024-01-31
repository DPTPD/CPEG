import numpy as np
import zfpy

from HoloUtils import getComplex
from methods.general.compressor import Compressor, HoloSpec


class OldZfp(Compressor):
    def __init__(self, rate):
        self.rate = rate

    def compress(self, hologram: HoloSpec, output_path: str) -> None:
        holo = hologram.holo
        imag_matrix = np.imag(holo)
        real_matrix = np.real(holo)

        compressed_data_imag = zfpy.compress_numpy(imag_matrix, rate=self.rate)
        compressed_data_real = zfpy.compress_numpy(real_matrix, rate=self.rate)

        with open(output_path + 'immaginaria_C.bin', 'wb') as f:
            f.write(compressed_data_imag)
        with open(output_path + 'reale_C.bin', 'wb') as f:
            f.write(compressed_data_real)

    def decompress(self, input_path: str) -> HoloSpec:
        with open(input_path + 'immaginaria_C.bin', 'rb') as f:
            compressed_data_imag = f.read()
        with open(input_path + 'reale_C.bin', 'rb') as f:
            compressed_data_real = f.read()

        decompressed_array_imag = zfpy.decompress_numpy(compressed_data_imag)
        decompressed_array_real = zfpy.decompress_numpy(compressed_data_real)

        complex_matrix = getComplex(decompressed_array_real, decompressed_array_imag)
        return HoloSpec(complex_matrix, 0, 0, 0)

    def is_lossless(self) -> bool:
        return False
