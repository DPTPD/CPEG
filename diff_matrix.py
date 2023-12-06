import main
from methods.general.compressor import HoloSpec
import numpy as np
import zfp_test

AXIS = 0


def main_func():
    mat: HoloSpec = main.open_hologram('mat_files/Hol_2D_dice.mat')
    hologram: np.ndarray = mat.holo
    # hologram = np.array([[1, 2, 6], [4, 5, 99]])
    diff = np.diff(hologram, n=1, axis=AXIS, prepend=np.zeros([1, hologram.shape[1]]))
    real, imag = zfp_test.compress_fpzip(diff)
    print(len(real) + len(imag))
    real, imag = zfp_test.compress_fpzip(hologram)
    print(len(real) + len(imag))
    pass


if __name__ == '__main__':
    main_func()
