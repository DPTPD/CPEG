import os.path
from abc import ABC, abstractmethod

import numpy as np


class HoloSpec:
    def __init__(self, holo: np.ndarray, pp: float, wlen: float, dist: float):
        self.holo = holo
        self.pp = pp
        self.wlen = wlen
        self.dist = dist

    def __eq__(self, other):
        if not isinstance(other, HoloSpec):
            return False
        return ((self.holo == other.holo).all() and
                self.pp == other.pp and
                self.wlen == other.wlen and
                self.dist == other.dist)


class Compressor(ABC):

    @abstractmethod
    def compress(self, hologram: HoloSpec, output_path: str) -> None:
        pass

    @abstractmethod
    def decompress(self, input_path: str) -> HoloSpec:
        pass

    @staticmethod
    def calculate_ratio(compressed_path: str, uncompressed_path: str) -> float:
        compressed_len = os.path.getsize(compressed_path)
        uncompressed_len = os.path.getsize(uncompressed_path)
        ratio = uncompressed_len / compressed_len
        return ratio
