import os

import numpy as np
import pywt

from methods.general.compressor import Compressor, HoloSpec


class OldWavelet(Compressor):
    def __init__(self, wavelet, mode, value):
        self.wavelet = wavelet
        self.value = value
        self.mode = mode

    def compress(self, hologram: HoloSpec, output_path: str) -> None:
        hologram = hologram.holo
        np.savez(output_path, hologram)
        coefficients = pywt.wavedec2(hologram, wavelet=self.wavelet)

        # Apply lossy compression to the wavelet coefficients
        for coeffs in coefficients:
            for i in range(len(coeffs)):
                for j in range(len(coeffs[i])):
                    coeffs[i][j] = pywt.threshold(coeffs[i][j], value=self.value, mode=self.mode)

        coefficients = np.array(coefficients, dtype='object')
        np.savez_compressed(output_path, coefficients)

    def decompress(self, input_path: str) -> HoloSpec:
        with np.load(input_path, allow_pickle=True) as data:
            coefficients = data['arr_0']

        coefficients = coefficients.tolist()
        compressed_hologram = pywt.waverec2(coefficients, wavelet=self.wavelet)
        return HoloSpec(compressed_hologram, 0, 0, 0)

    def is_lossless(self) -> bool:
        return False
