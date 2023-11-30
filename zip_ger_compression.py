import os
import numpy as np
import zlib
import pickle
import lossless_ger

from HoloUtils import getComplex, sizeof_fmt, rate


def zip_ger_compression(holo, filename):
    matrix = lossless_ger.func6(holo)
    compressed_data = zlib.compress(matrix)
    with open(filename, 'wb') as f:
        f.write(compressed_data)


def zip_ger_decompression(filename):
    with open(filename, 'rb') as f:
        compressed_data = f.read()
        data = zlib.decompress(compressed_data)













