import abc
import pickle
import typing

import methods.general.compressor
from methods.general.compressor import HoloSpec
from methods.general.floatifier import Floatifier
from methods.general.g17_transform import G17Transformer


class GeneralCompressor(methods.general.compressor.Compressor, abc.ABC):

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

    @abc.abstractmethod
    def compress_bytes(self, data: bytes) -> bytes:
        pass

    @abc.abstractmethod
    def decompress_bytes(self, data: bytes) -> bytes:
        pass

    def compress(self, hologram: HoloSpec, output_path: str):
        holo = hologram.holo
        floatified_holo = self.floatifier.apply(holo)
        g17ed_matrix = self.g17.apply(floatified_holo)
        data = pickle.dumps((g17ed_matrix, hologram.pp, hologram.wlen, hologram.dist))
        compressed = self.compress_bytes(data)
        with open(output_path, 'wb') as f:
            f.write(compressed)

    def decompress(self, input_path: str) -> HoloSpec:
        with open(input_path, 'rb') as f:
            c = self.decompress_bytes(f.read())
        compressed_data = pickle.loads(c)
        g17ed_matrix, pp, wlen, dist = compressed_data
        floatified_holo = self.g17.deapply(g17ed_matrix)
        holo = self.floatifier.deapply(floatified_holo)
        return HoloSpec(holo, pp, wlen, dist)
