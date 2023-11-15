import os
import struct

import cv2
import matplotlib.image as plt
import numpy as np
from PIL import Image

import compressor
from compressor import HoloSpec


class PNGCompressor(compressor.Compressor):
    @staticmethod
    def _reinterpret_as_f64(arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        arr = arr.reshape(-1, shape[-1])
        int_arr = np.array([struct.unpack("<f", struct.pack("BBBB", x[2], x[1], x[0], x[3])) for x in arr],
                           dtype=np.dtype("<f"))
        int_arr = int_arr.reshape(shape[:-1])
        return int_arr

    @staticmethod
    def _reinterpret_as_uint(arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        arr = arr.flatten()
        int_arr = np.array([struct.unpack("BBBB", struct.pack("<f", x)) for x in arr], dtype=np.dtype("B"))
        int_arr = int_arr.reshape((*shape, 4))
        return int_arr

    @staticmethod
    def _merge_matrices(matrix1, matrix2):
        if matrix1.shape[0] != matrix2.shape[0]:
            raise ValueError("Matrices must have the same number of rows.")

        return np.hstack((matrix1, matrix2))

    @staticmethod
    def _split_matrix(combined_matrix, split_point):
        if split_point < 0 or split_point >= combined_matrix.shape[1]:
            raise ValueError("Invalid split point.")

        matrix1 = combined_matrix[:, :split_point]
        matrix2 = combined_matrix[:, split_point:]

        return matrix1, matrix2

    @staticmethod
    def _double_to_ulong(double: float) -> int:
        return struct.unpack("<Q", struct.pack("d", double))[0]

    @staticmethod
    def _ulong_to_double(ulong: int) -> float:
        return struct.unpack("d", struct.pack("<Q", ulong))[0]

    def compress(self, hologram: HoloSpec, output_path: str):
        real_matrix = PNGCompressor._reinterpret_as_uint(np.real(hologram.holo))
        imag_matrix = PNGCompressor._reinterpret_as_uint(np.imag(hologram.holo))
        merged = PNGCompressor._merge_matrices(real_matrix, imag_matrix)
        metadata = {
            "pp": str(PNGCompressor._double_to_ulong(hologram.pp)),
            "wlen": str(PNGCompressor._double_to_ulong(hologram.wlen)),
            "dist": str(PNGCompressor._double_to_ulong(hologram.dist)),
        }
        plt.imsave(output_path, merged, metadata=metadata, format="png", pil_kwargs={"optimize": True})

    def decompress(self, input_path: str) -> HoloSpec:
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        pil_image = Image.open(input_path)
        metadata = pil_image.info
        pp = PNGCompressor._ulong_to_double(int(metadata["pp"]))
        wlen = PNGCompressor._ulong_to_double(int(metadata["wlen"]))
        dist = PNGCompressor._ulong_to_double(int(metadata["dist"]))
        img_real = PNGCompressor._reinterpret_as_f64(img)
        img_real, img_imag = PNGCompressor._split_matrix(img_real, img_real.shape[1] // 2)
        complex_matrix = np.stack([img_real, img_imag], axis=-1).view(np.complex64)
        complex_matrix = complex_matrix.reshape(complex_matrix.shape[:-1])
        return HoloSpec(complex_matrix, pp, wlen, dist)


def _get_names(directory: str, name: str) -> (str, str):
    name_imag = os.path.join(directory, name + "_img.png")
    name_real = os.path.join(directory, name + "_real.png")
    return name_real, name_imag
