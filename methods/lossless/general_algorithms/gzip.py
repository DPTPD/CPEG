import gzip

from methods.lossless.general_algorithms.general_compressor import GeneralCompressor


class GzipCompressor(GeneralCompressor):
    def compress_bytes(self, data: bytes) -> bytes:
        return gzip.compress(data, compresslevel=9)

    def decompress_bytes(self, data: bytes) -> bytes:
        return gzip.decompress(data)
