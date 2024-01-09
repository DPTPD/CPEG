import typing

import cv2

from methods.general.floatifier import Floatifier
from methods.general.g17_transform import G17Transformer
from methods.lossless.image_algorithms.image_compressor import PillowCompressor


# todo: WEBP a volte non è lossless con quality>100 ma non ha senso
class WebpCompressor(PillowCompressor):

    def is_lossless(self) -> bool:
        return self.quality > 100

    def __init__(self, floatifier: Floatifier | typing.Literal["in_place", "hstack", "vstack"],
                 g17: G17Transformer | typing.Literal[None, "bits", "bytes"], grayscale: bool = False, quality=101):
        super().__init__(floatifier, g17, [cv2.IMWRITE_WEBP_QUALITY, quality], "webp", grayscale)
        self.quality = quality
