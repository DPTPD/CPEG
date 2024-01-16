import os
import pickle
import subprocess
import tempfile
import typing

import cv2
import numpy as np

from methods.general.floatifier import Floatifier
from methods.general.g17_transform import G17Transformer
from methods.lossless.image_algorithms import PngCompressor
from methods.lossless.image_algorithms.image_compressor import ImageCompressor


class WebpCompressor(ImageCompressor):
    def decompress_image(self, input_path: str) -> (np.ndarray, float, float, float, str, int, int):
        with open(input_path, "rb") as fp:
            pp, wlen, dist, mat_dtype, webp_image, shape_0, dtype_size = pickle.load(fp)
        with tempfile.NamedTemporaryFile("wb", suffix=".webp") as fp:
            temp_name = fp.name
        with open(temp_name, "wb") as fp:
            fp.write(webp_image)
        img = cv2.imread(temp_name, cv2.IMREAD_UNCHANGED)
        os.remove(temp_name)
        if not self.png.grayscale:
            img = img.reshape((img.shape[0], -1))
        return img, pp, wlen, dist, mat_dtype, shape_0, dtype_size

    def compress_image(self, matrix: np.ndarray, pp: float, wlen: float, dist: float, mat_dtype: str, shape_0: int, dtype_size: int, output_path: str):
        with tempfile.NamedTemporaryFile("wb", suffix=".png") as fp:
            png_temp_name = fp.name
        self.png.compress_image(matrix, pp, wlen, dist, mat_dtype, shape_0,dtype_size, png_temp_name)
        if self.is_lossless():
            args = f"cwebp -lossless -exact {png_temp_name} -o".split(" ")
        else:
            args = f"cwebp -q {self.quality} {png_temp_name} -o".split(" ")
        args.append(output_path)
        subprocess.run(args, stderr=subprocess.DEVNULL)
        os.remove(png_temp_name)
        with open(output_path, "rb") as fp:
            data = fp.read()
        with open(output_path, "wb") as fp:
            pickle.dump((pp, wlen, dist, mat_dtype, data, shape_0, dtype_size), fp)

    def is_lossless(self) -> bool:
        return self.quality > 100

    def __init__(self, floatifier: Floatifier | typing.Literal["in_place", "hstack", "vstack"],
                 g17: G17Transformer | typing.Literal[None, "bits", "bytes"], grayscale: bool = False, quality=101):
        super().__init__(floatifier, g17)
        self.png = PngCompressor(floatifier, g17, grayscale)
        self.format_name = "webp"
        self.quality = quality
