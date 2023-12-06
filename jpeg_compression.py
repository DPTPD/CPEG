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

    @staticmethod
    def _merge_matrices(matrix1, matrix2):
        if matrix1.shape[0] != matrix2.shape[0]:
            raise ValueError("Matrices must have the same number of rows.")

        return np.hstack((matrix1, matrix2))

    @staticmethod
    def _split_matrix(combined_matrix, split_point):
        if split_point < 0 or split_point >= combined_matrix.shape[1]:
            raise ValueError("Invalid split point.")

        matrix1 = combined_matrix[:, :split_point]
        matrix2 = combined_matrix[:, split_point:]

        return matrix1, matrix2

    def compress(self, hologram: HoloSpec, output_path: str) -> None:
        imagMatrix = np.imag(hologram.holo)
        realMatrix = np.real(hologram.holo)
        l = JPEGCompressor._merge_matrices(realMatrix, imagMatrix)
        # JPG
        matplotlib.image.imsave(output_path, l, cmap='gray')

    def decompress(self, input_path: str) -> HoloSpec:
        img_real = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        img_real, img_imag = JPEGCompressor._split_matrix(img_real, img_real.shape[1] // 2)
        return HoloUtils.getComplex(img_real, img_imag)


def main2():
    holoFileName = 'mat_files/Hol_2D_dice.mat'
    x = main.open_hologram(holoFileName)
    JPEGCompressor().compress(x, "jpegtest.jpg")
    newHolo = JPEGCompressor().decompress("jpegtest.jpg")
    similarity = paper_similarity.Similarity(paper_similarity.GammaM.bump, paper_similarity.GammaR.cos,
                                             paper_similarity.GammaA.unique)
    print(similarity.calc_similarity(x.holo, newHolo))
    # 1.7090396233367385e-10


if __name__ == '__main__':
    main2()
