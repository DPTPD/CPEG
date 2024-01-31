import typing

import cv2

from methods.general.floatifier import Floatifier
from methods.general.g17_transform import G17Transformer
from methods.lossless.image_algorithms.image_compressor import PillowCompressor


class JpegCompressor(PillowCompressor):
    def support_transparency(self) -> bool:
        return False

    def is_lossless(self) -> bool:
        return False

    def __init__(self, floatifier: Floatifier | typing.Literal["in_place", "hstack", "vstack"],
                 g17: G17Transformer | typing.Literal[None, "bits", "bytes"],
                 grayscale: bool = False,
                 output_quality: int = 95
                 ):
        super().__init__(floatifier, g17, [cv2.IMWRITE_JPEG_QUALITY, output_quality], "jpeg", grayscale)
