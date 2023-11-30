import numpy
import numpy as np
import struct

a = np.array([[1., 2., 3.], [4., 5., 6.]])


# %%
def func(val, n_bits_component):
    val_real, val_imag = struct.unpack("LL", struct.pack("dd", val.real, val.imag))
    arr = [False] * (n_bits_component*2)

    for i in range(n_bits_component):
        arr[i] = val_real & (1 << (n_bits_component - i - 1)) != 0
    for i in range(n_bits_component):
        arr[i + n_bits_component] = val_imag & (1 << (n_bits_component - i - 1)) != 0
    return arr


def func2(val):
    return func(val, 64)


def func3(arr):
    sh = arr.shape
    arr = arr.flatten()
    arr2 = []
    for x in np.nditer(arr):
        arr2.extend(func2(x))

    arr2 = numpy.array(arr2)
    return arr2.reshape((*sh[:-1], -1))


def func4(arr):
    sh = arr.shape
    arr = np.transpose(arr)
    temp = arr.reshape((-1, 8))
    temp2 = np.packbits(temp)
    # x = [struct.pack("B", x) for x in temp2]
    return temp2.tobytes()


def func6(a):
    test = func3(a)
    return func4(test)
