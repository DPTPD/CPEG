import typing

import cv2

from methods.general.floatifier import Floatifier
from methods.general.g17_transform import G17Transformer
from methods.lossless.image_algorithms.image_compressor import PillowCompressor


class PngCompressor(PillowCompressor):

    def __init__(self, floatifier: Floatifier | typing.Literal["in_place", "hstack", "vstack"],
                 g17: G17Transformer | typing.Literal[None, "bits", "bytes"],
                 grayscale: bool = False):
        super().__init__(floatifier, g17, [cv2.IMWRITE_PNG_COMPRESSION, 9], "png", grayscale)
