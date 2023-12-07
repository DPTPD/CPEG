import pickle

import methods.general.compressor
import methods.general.zfp_utils
from methods.general.compressor import HoloSpec


class Zfp(methods.general.compressor.Compressor):

    def __init__(self, precision: int):
        self.precision = precision

    def compress(self, hologram: HoloSpec, output_path: str) -> None:
        data_real, data_imag = methods.general.zfp_utils.compress_fpzip(hologram.holo, self.precision)
        with open(output_path, "wb") as fp:
            pickle.dump((data_real, data_imag, hologram.pp, hologram.wlen, hologram.dist), fp)

    def decompress(self, input_path: str) -> HoloSpec:
        with open(input_path, "rb") as fp:
            data_real, data_imag, pp, wlen, dist = pickle.load(fp)
        matrix = methods.general.zfp_utils.decompress_fpzip(data_real, data_imag)
        return methods.general.compressor.HoloSpec(matrix, pp, wlen, dist)
