import gzip
import pickle
import numpy as np

import HoloUtils
from methods.general.compressor import HoloSpec, Compressor
from methods.lossy.jpeg import JPEGCompressor

dict_name = 'gzip_compression/'


class GzipCompressor(Compressor):
    def compress(self, hologram: HoloSpec, output_path: str) -> None:
        pp = hologram.pp
        wlen = hologram.wlen
        holo = hologram.holo
        dist = hologram.dist

        imag_matrix = np.imag(holo)
        real_matrix = np.real(holo)
        matrix = JPEGCompressor._merge_matrices(real_matrix, imag_matrix)

        compressed_holo = gzip.compress(pickle.dumps(matrix))
        compressed_data = (compressed_holo, pp, wlen, dist)
        with open(output_path, 'wb') as f:
            pickle.dump(compressed_data, f) # 30,9MB

    def decompress(self, input_path: str) -> HoloSpec:
        with open(input_path, 'rb') as f:
            compressed_data = pickle.load(f)

        compressed_holo, pp, wlen, dist = compressed_data
        matrix = gzip.decompress(compressed_holo)
        matrix = pickle.loads(matrix)
        real_matrix, imag_matrix = JPEGCompressor._split_matrix(matrix, matrix.shape[1] // 2)

        holo = HoloUtils.getComplex(real_matrix, imag_matrix)
        return HoloSpec(holo, pp, wlen, dist)
