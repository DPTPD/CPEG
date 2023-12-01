import main
import zfpy

holoFileName = 'mat_files/Hol_2D_dice.mat'
x = main.open_hologram(holoFileName).holo
compressed_data_real = zfpy.compress_numpy(x.real,precision=100)
compressed_data_imag = zfpy.compress_numpy(x.imag)
decompressed_array_real = zfpy.decompress_numpy(compressed_data_real)
decompressed_array_imag = zfpy.decompress_numpy(compressed_data_imag)
with open("zfp_test.out", "wb") as fp:
    fp.write(compressed_data_real)
    fp.write(compressed_data_imag)

pass
