import bz2
import pickle

from methods.general.compressor import HoloSpec, Compressor
from methods.lossless.lossless_ger import apply_ger_transformation

dict_name = 'bzip2_compression/'


class Bzip2GerCompressor(Compressor):
    def compress(self, hologram: HoloSpec, output_path: str) -> None:
        pp = hologram.pp
        wlen = hologram.wlen
        holo = hologram.holo
        dist = hologram.dist

        matrix = apply_ger_transformation(holo)

        compressed_holo = bz2.compress(pickle.dumps(matrix))
        compressed_data = (compressed_holo, pp, wlen, dist)
        with open(output_path, 'wb') as f:
            pickle.dump(compressed_data, f)

    def decompress(self, input_path: str) -> HoloSpec:
        raise NotImplementedError
