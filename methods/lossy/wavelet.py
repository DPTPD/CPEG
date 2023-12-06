import numpy as np
import pywt

import main
from methods.general import paper_similarity, compressor
from methods.general.compressor import HoloSpec

dict_name = 'wavelet_compression/'


class WaveletCompressor(compressor.Compressor):

    def compress(self, hologram: HoloSpec, output_path: str) -> None:
        wavelet = 'db4'
        mode = 'hard'
        value = 200000000
        coefficients = pywt.wavedec2(hologram.holo, wavelet=wavelet)

        # Apply lossy compression to the wavelet coefficients
        for coeffs in coefficients:
            for i in range(len(coeffs)):
                for j in range(len(coeffs[i])):
                    coeffs[i][j] = pywt.threshold(coeffs[i][j], value=value, mode=mode)

        coefficients = np.array(coefficients, dtype='object')
        np.savez_compressed(output_path, coefficients)

    def decompress(self, input_path: str) -> HoloSpec:
        wavelet = 'db4'
        with np.load(input_path, allow_pickle=True) as data:
            # ottieni tutti gli array presenti nel file
            coefficients = data['arr_0']
        # Reconstruct the compressed hologram data
        coefficients = coefficients.tolist()
        return pywt.waverec2(coefficients, wavelet=wavelet)


def main2():
    holoFileName = 'mat_files/Hol_2D_dice.mat'
    x = main.open_hologram(holoFileName)
    WaveletCompressor().compress(x, "../../jpegtest.npz")
    newHolo = WaveletCompressor().decompress("../../jpegtest.npz")
    similarity = paper_similarity.Similarity(paper_similarity.GammaM.bump, paper_similarity.GammaR.cos,
                                             paper_similarity.GammaA.unique)
    print(similarity.calc_similarity(x.holo, newHolo))
    # 0.4423582261773792


if __name__ == '__main__':
    main2()
