import bz2

from methods.lossless.general_algorithms.general_compressor import GeneralCompressor


class Bzip2Compressor(GeneralCompressor):

    def compress_bytes(self, data: bytes) -> bytes:
        return bz2.compress(data, compresslevel=9)

    def decompress_bytes(self, data: bytes) -> bytes:
        return bz2.decompress(data)
