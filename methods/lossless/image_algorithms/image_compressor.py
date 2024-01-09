import abc
import json
import struct
import typing

import cv2
import numpy as np
from PIL import Image

import methods.general.compressor
from methods.general.compressor import HoloSpec
from methods.general.floatifier import Floatifier
from methods.general.g17_transform import G17Transformer


class ImageCompressor(methods.general.compressor.Compressor, abc.ABC):

    def __init__(self, floatifier: Floatifier | typing.Literal["in_place", "hstack", "vstack"],
                 g17: G17Transformer | typing.Literal[None, "bits", "bytes"]):
        if isinstance(floatifier, Floatifier):
            self.floatifier = floatifier
        else:
            self.floatifier = Floatifier(floatifier)

        if isinstance(g17, G17Transformer):
            self.g17 = g17
        else:
            self.g17 = G17Transformer(g17)

    @staticmethod
    def _add_metadata_to_image(metadata_dict, image_path, output_path):
        try:
            # Open the WebP image
            with Image.open(image_path) as img:
                metadata_bytes = bytes(json.dumps(metadata_dict), "utf-8")
                # Convert metadata to bytes
                # Add metadata using TextChunk
                img.save(output_path, img.format, exif=metadata_bytes, lossless=True)
        except Exception as e:
            print(f"Error: {e}")

    @staticmethod
    def _double_to_ulong(double: float) -> int:
        return struct.unpack("<Q", struct.pack("d", double))[0]

    @staticmethod
    def _ulong_to_double(ulong: int) -> float:
        return struct.unpack("d", struct.pack("<Q", ulong))[0]

    @abc.abstractmethod
    def compress_image(self, matrix: np.ndarray, pp: float, wlen: float, dist: float, mat_dtype: str, output_path: str):
        """
        :param matrix: a u8 matrix
        """
        pass

    @abc.abstractmethod
    def decompress_image(self, input_path: str) -> (np.ndarray, float, float, float, str):
        """
        :return: (u8 matrix, pp, wlen, dist,mat_dtype)
        """
        pass

    def compress(self, hologram: HoloSpec, output_path: str):
        holo = hologram.holo
        floatified_holo = self.floatifier.apply(holo)
        g17ed_matrix = self.g17.apply(floatified_holo)
        mat_dtype = g17ed_matrix.dtype
        g17ed_matrix = g17ed_matrix.view(np.uint8).copy()
        self.compress_image(g17ed_matrix, hologram.pp, hologram.wlen, hologram.dist, mat_dtype.name, output_path)

    def decompress(self, input_path: str) -> HoloSpec:
        g17ed_matrix, pp, wlen, dist, mat_dtype = self.decompress_image(input_path)
        g17ed_matrix = g17ed_matrix.view(np.dtype(mat_dtype)).copy()
        floatified_holo = self.g17.deapply(g17ed_matrix)
        holo = self.floatifier.deapply(floatified_holo)
        return HoloSpec(holo, pp, wlen, dist)


class PillowCompressor(ImageCompressor, abc.ABC):
    def __init__(self, floatifier: Floatifier | typing.Literal["in_place", "hstack", "vstack"],
                 g17: G17Transformer | typing.Literal[None, "bits", "bytes"],
                 params: list,
                 format_name: str,
                 grayscale: bool = False):
        super().__init__(floatifier, g17)
        self.grayscale = grayscale
        self.format_name = format_name
        self.params = params

    def compress_image(self, matrix: np.ndarray, pp: float, wlen: float, dist: float, mat_dtype: str, output_path: str):
        merged = matrix
        if not self.grayscale:
            merged = merged.reshape((merged.shape[0], -1, 4))
        metadata = {
            "pp": str(ImageCompressor._double_to_ulong(pp)),
            "wlen": str(ImageCompressor._double_to_ulong(wlen)),
            "dist": str(ImageCompressor._double_to_ulong(dist)),
            "dtype": mat_dtype
        }
        cv2.imwrite(output_path, merged, params=self.params)
        ImageCompressor._add_metadata_to_image(metadata, output_path, output_path)

    def decompress_image(self, input_path: str) -> (np.ndarray, float, float, float, str):
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if not self.grayscale:
            img = img.reshape((img.shape[0], -1))
        pil_image = Image.open(input_path)
        metadata: bytes = pil_image.info.get("exif")
        if metadata is None:
            metadata = pil_image.app["APP1"]
        if metadata.startswith(b"Exif\x00\x00"):
            metadata = metadata[6:]
        metadata: str = metadata.decode("utf-8")
        metadata: dict = json.loads(metadata)
        dtype = metadata["dtype"]
        pp = ImageCompressor._ulong_to_double(int(metadata["pp"]))
        wlen = ImageCompressor._ulong_to_double(int(metadata["wlen"]))
        dist = ImageCompressor._ulong_to_double(int(metadata["dist"]))
        return img, pp, wlen, dist, dtype
