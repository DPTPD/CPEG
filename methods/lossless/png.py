import os
import struct

import PIL.PngImagePlugin
import cv2
import numpy as np
from PIL import Image

from methods.general import compressor
from methods.general.compressor import HoloSpec


class PNGCompressor(compressor.Compressor):
    @staticmethod
    def _reinterpret_as_complex(arr: np.ndarray, byt) -> np.ndarray:
        shape = arr.shape
        dtype = np.dtype('c' + str(byt))
        arr = arr.reshape((shape[0], -1, byt))
        arr = arr.view(dtype)
        arr = arr.reshape((arr.shape[0], arr.shape[1]))
        return arr

    @staticmethod
    def _reinterpret_as_uint(arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        arr = arr.flatten()
        int_arr = arr.view(np.uint8)
        int_arr = int_arr.reshape((shape[0], -1, 4))

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

    def _add_metadata_to_image(self, metadata_dict, image_path, output_path):
        # Open the image
        img = Image.open(image_path)

        # Get existing metadata (if any)
        info = PIL.PngImagePlugin.PngInfo()
        for key, value in metadata_dict.items():
            info.add_text(key, value)

        # Save the modified image with updated metadata
        img.save(output_path, format=img.format, pnginfo=info)

    def compress(self, hologram: HoloSpec, output_path: str):
        merged = PNGCompressor._reinterpret_as_uint(hologram.holo)
        metadata = {
            "pp": str(PNGCompressor._double_to_ulong(hologram.pp)),
            "wlen": str(PNGCompressor._double_to_ulong(hologram.wlen)),
            "dist": str(PNGCompressor._double_to_ulong(hologram.dist)),
            "bytes": str(hologram.holo.dtype.itemsize)
        }
        cv2.imwrite(output_path, merged, params=[cv2.IMWRITE_PNG_COMPRESSION, 9])
        self._add_metadata_to_image(metadata, output_path, output_path)

    def decompress(self, input_path: str) -> HoloSpec:
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        pil_image = Image.open(input_path)
        metadata = pil_image.info
        byt = int(metadata["bytes"])
        pp = PNGCompressor._ulong_to_double(int(metadata["pp"]))
        wlen = PNGCompressor._ulong_to_double(int(metadata["wlen"]))
        dist = PNGCompressor._ulong_to_double(int(metadata["dist"]))
        img_real = PNGCompressor._reinterpret_as_complex(img, byt)
        return HoloSpec(img_real, pp, wlen, dist)


def _get_names(directory: str, name: str) -> (str, str):
    name_imag = os.path.join(directory, name + "_img.png")
    name_real = os.path.join(directory, name + "_real.png")
    return name_real, name_imag
