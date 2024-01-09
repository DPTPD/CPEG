import pickle
import typing
from abc import ABC, abstractmethod

import numpy as np

from methods.general.compressor import Compressor, HoloSpec
from methods.general.floatifier import Floatifier
from methods.general.g17_transform import G17Transformer


class FpAlgorithm(Compressor, ABC):
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
        compressed = self.compress_float_matrix(holo)
        data = pickle.dumps((compressed, hologram.pp, hologram.wlen, hologram.dist))
        with open(output_path, 'wb') as f:
            f.write(data)

    def decompress(self, input_path: str) -> HoloSpec:
        with open(input_path, 'rb') as f:
            compressed, pp, wlen, dist = pickle.load(f)
        holo = self.decompress_float_matrix(compressed)
        return HoloSpec(holo, pp, wlen, dist)

    def compress_float_matrix(self, matrix: np.ndarray) -> bytes:
        floatified_holo = self.floatifier.apply(matrix)
        g17ed_matrix = self.g17.apply(floatified_holo)
        shape = g17ed_matrix.shape
        reshaped_matrix = g17ed_matrix.reshape((-1,))
        final = self._compress_float_matrix(reshaped_matrix)
        return pickle.dumps((shape, final))

    def decompress_float_matrix(self, data: bytes) -> np.ndarray:
        shape, final = pickle.loads(data)
        reshaped_matrix = self._decompress_float_matrix(final)
        g17ed_matrix = reshaped_matrix.reshape(shape)
        floatified_holo = self.g17.deapply(g17ed_matrix)
        final = self.floatifier.deapply(floatified_holo)
        return final

    @abstractmethod
    def _compress_float_matrix(self, matrix: np.ndarray) -> bytes:
        pass

    @abstractmethod
    def _decompress_float_matrix(self, data: bytes) -> np.ndarray:
        pass
