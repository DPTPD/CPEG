import numpy
import pyPeriod.QOPeriods
import numpy as np
import fpzip
import main


def inverse(A: np.ndarray) -> np.ndarray:
    return np.linalg.inv(A)


phi_1080 = {'1': 1, '2': 1, '3': 2, '4': 2, '5': 4, '6': 2, '8': 4, '9': 6, '10': 4, '12': 4, '15': 8, '18': 6, '20': 8,
            '24': 8, '27': 18, '30': 8, '36': 12, '40': 16, '45': 24, '54': 18, '60': 16, '72': 24, '90': 24, '108': 36,
            '120': 32, '135': 72, '180': 48, '216': 72, '270': 72, '360': 96, '540': 144, '1080': 288}
phi_1920 = {'1': 1, '2': 1, '3': 2, '4': 2, '5': 4, '6': 2, '8': 4, '10': 4, '12': 4, '15': 8, '16': 8, '20': 8,
            '24': 8,
            '30': 8, '32': 16, '40': 16, '48': 16, '60': 16, '64': 32, '80': 32, '96': 32, '120': 32, '128': 64,
            '160': 64,
            '192': 64, '240': 64, '320': 128, '384': 128, '480': 128, '640': 256, '960': 256, '1920': 512}
phi_1087 = {'1': 1, '1087': 1086}
phi_1931 = {'1': 1, '1931': 1930}

P_N = numpy.load("ramanujan/1087.npy")
P_M = numpy.load("ramanujan/1931.npy")


holoFileName = 'mat_files/Hol_2D_dice.mat'


def transform(x):
    return np.matmul(np.matmul(inverse(P_N), x), np.transpose(inverse(P_M)))


x = pyPeriod.RamanujanPeriods(basis_type="ramanujan")

x = main.open_hologram(holoFileName).holo

matrix_20x20 = np.zeros((1087, 1931),dtype=np.complex128)

# Copy the elements from the 10x10 matrix to the top-left corner of the 20x20 matrix
matrix_20x20[:1080, :1920] = x

x=matrix_20x20

compressed_bytes_real = fpzip.compress(x.real, precision=0, order='F')  # returns byte string
with open("ramanujan/c_total_real", "wb") as fp:
    fp.write(compressed_bytes_real)
compressed_bytes_imag = fpzip.compress(x.imag, precision=0, order='F')  # returns byte string
with open("ramanujan/c_total_imag", "wb") as fp:
    fp.write(compressed_bytes_imag)

with open("ramanujan/c_total_imag", "rb") as fp:
    x_imag = fpzip.decompress(fp.read(), order='F')
with open("ramanujan/c_total_real", "rb") as fp:
    x_real = fpzip.decompress(fp.read(), order='F')

complex_matrix = np.stack([x_real, x_imag], axis=-1).view(np.complex128)
complex_matrix = complex_matrix.reshape((1087, 1931))
assert (x == complex_matrix).all()

final = transform(x)
final_real = transform(x.real)
final_imag = transform(x.imag)
compressed_bytes_real = fpzip.compress(final_real, precision=0, order='C')  # returns byte string
compressed_bytes_imag = fpzip.compress(final_imag, precision=0, order='C')  # returns byte string
with open("ramanujan/c_real", "wb") as fp:
    fp.write(compressed_bytes_real)
with open("ramanujan/c_imag", "wb") as fp:
    fp.write(compressed_bytes_imag)
print(final)
pass
