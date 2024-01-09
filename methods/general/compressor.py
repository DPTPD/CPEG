from abc import ABC, abstractmethod

import numpy as np
import scipy


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

    @staticmethod
    def open_hologram(path: str) -> 'HoloSpec':
        f = scipy.io.loadmat(path)  # aprire il file .mat
        # Per comprimere
        pp = f['pitch'][0][0]  # pixel pitch
        wlen = f['wlen'][0][0]  # wavelenght
        dist = f['zobj'][0][0]  # propogation depth
        # Per renderizzare
        # pp = np.matrix(f['pitch'][0])  # pixel pitch
        # wlen = np.matrix(f['wlen'][0])  # wavelenght
        # dist = np.matrix(f['zobj1'][0])  # propogation depth

        holo = f['Hol']
        # holo = holo.astype(np.complex64)
        return HoloSpec(holo, pp, wlen, dist)


class CompareInfo:
    def __init__(self, ratio: float, similarity: float):
        self.ratio = ratio
        self.similarity = similarity


class Compressor(ABC):

    @abstractmethod
    def compress(self, hologram: HoloSpec, output_path: str) -> None:
        pass

    @abstractmethod
    def decompress(self, input_path: str) -> HoloSpec:
        pass

    @abstractmethod
    def is_lossless(self) -> bool:
        raise NotImplemented
