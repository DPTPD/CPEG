import lzma
import typing
from typing import Sequence, Mapping, Any

from methods.lossless.general_algorithms.general_compressor import GeneralCompressor


class LZMACompressor(GeneralCompressor):

    def __init__(self, floatifier: typing.Literal["in_place", "hstack", "vstack"],
                 g17: typing.Literal[None, "bits", "bytes"], filters: Sequence[Mapping[str, Any]] | None = None):
        super().__init__(floatifier, g17)
        self.filters = filters

    def compress_bytes(self, data: bytes) -> bytes:
        return lzma.compress(data, format=lzma.FORMAT_XZ, preset=lzma.PRESET_EXTREME, filters=self.filters)

    def decompress_bytes(self, data: bytes) -> bytes:
        return lzma.decompress(data)
