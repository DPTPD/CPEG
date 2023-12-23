import zlib

from methods.lossless.general_algorithms.general_compressor import GeneralCompressor


class ZipCompressor(GeneralCompressor):
    def compress_bytes(self, data: bytes) -> bytes:
        return zlib.compress(data, level=zlib.Z_BEST_COMPRESSION)

    def decompress_bytes(self, data: bytes) -> bytes:
        return zlib.decompress(data)
