import pickle

import numpy as np

import main
from methods.general.compressor import Compressor, HoloSpec
from methods.lossless.fp_algorithms.fp_algorithm import FpAlgorithm
from methods.lossless.fp_algorithms.zfp import ZfpCompressor


def _apply_low_pass_filter(image_fourier, cutoff_frequency):
    rows, cols = image_fourier.shape
    mask = np.ones((rows, cols), dtype=np.complex128)
    center_row, center_col = rows // 2, cols // 2
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            if distance > cutoff_frequency:
                mask[i, j] = 0
    filtered_image = image_fourier * mask

    return filtered_image


def _apply_high_pass_filter(image_fourier, threshold):
    rows, cols = image_fourier.shape
    mask = np.ones((rows, cols), dtype=np.complex128)
    center_row, center_col = rows // 2, cols // 2
    mask[center_row - threshold:center_row + threshold, center_col - threshold:center_col + threshold] = 0
    filtered_image = image_fourier * mask

    return filtered_image


class FourierCompressor(Compressor):
    def is_lossless(self) -> bool:
        return True

    def compress(self, hologram: HoloSpec, output_path: str) -> None:
        fourier_holo = np.fft.fftshift(np.fft.fft2(hologram.holo))
        filter_holo = self.pass_filter(fourier_holo, self.threshold)
        with open(output_path, "wb") as fp:
            a = (hologram.dist, hologram.wlen, hologram.pp,
                 self.float_compressor.compress_float_matrix(filter_holo.real),
                 self.float_compressor.compress_float_matrix(filter_holo.imag))
            pickle.dump(a, fp)

    def decompress(self, input_path: str) -> HoloSpec:
        with open(input_path, 'rb') as f:
            dist, wlen, pp, real_holo, imag_holo = pickle.load(f)
        real_holo = self.float_compressor.decompress_float_matrix(real_holo)
        imag_holo = self.float_compressor.decompress_float_matrix(imag_holo)
        holo = np.fft.ifft2(np.fft.ifftshift(real_holo + 1j * imag_holo))
        return HoloSpec(holo, pp, wlen, dist)

    def __init__(self, threshold: int, float_compressor: FpAlgorithm, use_low_pass_filter: bool = True):
        super().__init__()
        self.pass_filter = _apply_low_pass_filter if use_low_pass_filter else _apply_high_pass_filter
        self.threshold = threshold
        self.float_compressor = float_compressor


if __name__ == '__main__':
    mat: HoloSpec = main.open_hologram('mat_files/Hol_2D_dice.mat')
    compressor = FourierCompressor(10, ZfpCompressor("in_place", None, 0), True)
    compressor.compress(mat, "TODELETE.roba")
