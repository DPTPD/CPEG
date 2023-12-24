import os.path
import struct

import numpy
import numpy as np
import scipy

import HoloUtils
from methods.general import compressor

assert os.path.exists("mat_files"), "Missing mat_files folder in "+str(os.listdir("."))


def uncompress(input_file, output_file):
    f = scipy.io.loadmat(input_file)  # aprire il file .mat
    matrix = (f['Hol'])

    # Get the shape of the matrix
    shape = matrix.shape

    # Flatten the matrix to a 1D array
    flattened_matrix = matrix.flatten()

    # Open a file for writing in binary mode
    with open(output_file, 'wb') as file:
        # Write the shape information to the file
        file.write(struct.pack('<2I', *shape))  # '2I' means two unsigned integers

        for x in flattened_matrix:
            file.write(struct.pack('<2d', x.real, x.imag))


def open_hologram(path: str) -> compressor.HoloSpec:
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
    return compressor.HoloSpec(holo, pp, wlen, dist)


def extract_center_square(matrix: np.ndarray) -> np.ndarray:
    n, m = matrix.shape
    k = min(n, m)
    start_row = (n - k) // 2
    start_col = (m - k) // 2
    return matrix[start_row:start_row + k, start_col:start_col + k]


def render(holoFile: str):
    spec = open_hologram(holoFile)
    holo = extract_center_square(spec.holo)
    HoloUtils.hologramReconstruction(holo, spec.pp, spec.dist, spec.wlen)


