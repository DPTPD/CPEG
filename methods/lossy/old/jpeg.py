import cv2
import matplotlib
import numpy as np

from HoloUtils import getComplex
from methods.general.compressor import Compressor, HoloSpec


class OldJpeg(Compressor):
    def compress(self, hologram: HoloSpec, output_path: str) -> None:
        holo = hologram.holo
        imag_matrix = np.imag(holo)
        real_matrix = np.real(holo)
        name_real = output_path + '_reale.jpg'
        name_imag = output_path + '_immaginaria.jpg'
        # JPG
        matplotlib.image.imsave(name_real, real_matrix, cmap='gray')
        matplotlib.image.imsave(name_imag, imag_matrix, cmap='gray')

    def decompress(self, input_path: str) -> HoloSpec:
        name_real = input_path + '_reale.jpg'
        name_imag = input_path + '_immaginaria.jpg'
        img_real = cv2.imread(name_real, cv2.IMREAD_GRAYSCALE)
        img_imag = cv2.imread(name_imag, cv2.IMREAD_GRAYSCALE)
        complex_matrix = getComplex(img_real, img_imag)
        return HoloSpec(complex_matrix, 0, 0, 0)

    def is_lossless(self) -> bool:
        return False
