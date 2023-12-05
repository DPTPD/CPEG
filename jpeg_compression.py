import cv2
import matplotlib
import numpy as np

import HoloUtils
import compressor
import main
import paper_similarity
from compressor import HoloSpec

dict_name = 'image_based_compression/'


class JPEGCompressor(compressor.Compressor):

    def compress(self, hologram: HoloSpec, output_path: str) -> None:
        imagMatrix = np.imag(hologram.holo)
        realMatrix = np.real(hologram.holo)
        name_real = output_path+".real.jpg"
        name_imag = output_path+".imag.jpg"

        # JPG
        matplotlib.image.imsave(name_real, realMatrix, cmap='gray')
        matplotlib.image.imsave(name_imag, imagMatrix, cmap='gray')

    def decompress(self, input_path: str) -> HoloSpec:
        name_real = input_path+".real.jpg"
        name_imag = input_path+".imag.jpg"

        img_real = cv2.imread(name_real, cv2.IMREAD_GRAYSCALE)
        img_imag = cv2.imread(name_imag, cv2.IMREAD_GRAYSCALE)

        return HoloUtils.getComplex(img_real, img_imag)


def main2():
    holoFileName = 'mat_files/Hol_2D_dice.mat'
    x = main.open_hologram(holoFileName)
    JPEGCompressor().compress(x,"jpegtest")
    newHolo=JPEGCompressor().decompress("jpegtest")
    similarity=paper_similarity.Similarity(paper_similarity.GammaM.bump,paper_similarity.GammaR.cos,paper_similarity.GammaA.unique)
    print(similarity.calc_similarity(x.holo,newHolo))
    # 1.7090396233367385e-10

if __name__ == '__main__':
    main2()