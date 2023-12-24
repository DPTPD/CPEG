import pickle
import typing

import zfpy

import methods.general.compressor
import methods.general.zfp_utils
from methods.general.compressor import HoloSpec
from methods.general.floatifier import Floatifier
from methods.general.g17_transform import G17Transformer


class ZfpCompressor(methods.general.compressor.Compressor):

    def __init__(self, floatifier: Floatifier | typing.Literal["in_place", "hstack", "vstack"],
                 g17: G17Transformer | typing.Literal[None, "bytes"], precision: int):

        if g17 in [None, "bytes"]:
            g17 = G17Transformer(g17)
        if g17.g17 == "bits":
            raise ValueError

        if isinstance(floatifier, Floatifier):
            self.floatifier = floatifier
        else:
            self.floatifier = Floatifier(floatifier)

        self.g17 = g17
        self.precision = precision

    def compress(self, hologram: HoloSpec, output_path: str) -> None:
        holo = hologram.holo
        floatified_holo = self.floatifier.apply(holo)
        g17ed_matrix = self.g17.apply(floatified_holo)
        compressed = zfpy.compress_numpy(g17ed_matrix, precision=self.precision)
        data = pickle.dumps((compressed, hologram.pp, hologram.wlen, hologram.dist))
        with open(output_path, 'wb') as f:
            f.write(data)

    def decompress(self, input_path: str) -> HoloSpec:
        with open(input_path, 'rb') as f:
            compressed, pp, wlen, dist = pickle.load(f)
        g17ed_matrix = zfpy.decompress_numpy(compressed)
        floatified_holo = self.g17.deapply(g17ed_matrix)
        holo = self.floatifier.deapply(floatified_holo)
        return HoloSpec(holo, pp, wlen, dist)
