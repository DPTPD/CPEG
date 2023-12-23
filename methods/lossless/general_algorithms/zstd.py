import zstandard

from methods.lossless.general_algorithms.general_compressor import GeneralCompressor


class ZstdCompressor(GeneralCompressor):
    def compress_bytes(self, data: bytes) -> bytes:
        return zstandard.compress(data, level=zstandard.MAX_COMPRESSION_LEVEL)

    def decompress_bytes(self, data: bytes) -> bytes:
        return zstandard.decompress(data)
