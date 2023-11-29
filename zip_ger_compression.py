import os
import numpy as np
import zlib
import pickle
import lossless_ger

from HoloUtils import getComplex, sizeof_fmt, rate

dict_name = 'zip_ger_compression/'


def zip_ger_compression(holo, filename):
    print('ZIP_GER COMPRESSION ALGORITHM')
    if not os.path.isdir(dict_name + filename):
        os.makedirs(dict_name + filename)
    np.savez(dict_name + filename + '/matrix_HOLO', holo)

    matrix = lossless_ger.func6(holo)
    compressed_data = zlib.compress(matrix)

    with open(dict_name + filename + '/compresso.bin', 'wb') as f:
        f.write(compressed_data)

    # with open(dict_name + filename + '/matrix_dopotr.bin', 'wb') as f:
    #    f.write(matrix)

def zip_ger_decompression(filename):
    print('ZIP_GER DECOMPRESSION ALGORITHM')
    with open(dict_name + filename + '/compresso.bin', 'rb') as f:
        compressed_data = f.read()
        data = zlib.decompress(compressed_data)

        with open(dict_name + filename + '/decompresso.bin', 'wb') as f:
            f.write(data)

    total_size_HOL_NC = os.path.getsize(dict_name + filename + '/matrix_HOLO.npz')
    _ , total_size_HOL_P_formatted = sizeof_fmt(total_size_HOL_NC)
    print('NON COMPRESSA: ', total_size_HOL_P_formatted)

    total_size_HOL_C = os.path.getsize(dict_name + filename + '/compresso.bin')
    _ , total_size_HOL_P_formatted = sizeof_fmt(total_size_HOL_C)
    print('COMPRESSA: ', total_size_HOL_P_formatted)

    rate(total_size_HOL_NC, total_size_HOL_C)













