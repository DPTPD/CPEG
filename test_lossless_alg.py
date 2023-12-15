import os

import HoloUtils
from methods.lossless.bzip2ger_compression import Bzip2GerCompressor
from methods.general import paper_similarity
from methods.lossless.bzip2_compression import Bzip2Compressor
from methods.lossless.gzip_compression import GzipCompressor
from methods.lossless.gzipger_compression import GzipGerCompressor
from methods.lossless.zip_compression import ZipCompressor
from methods.lossless.zipger_compression import ZipGerCompressor
from methods.lossless.lzma_compression import LZMACompressor
from methods.lossless.lzmager_compression import LZMAGerCompressor
from methods.lossless.zstdger_compression import ZSTDGerCompressor
from methods.general.compressor import Compressor
from methods.general.compressor import HoloSpec
from methods.lossless.zstd_compression import ZSTDCompressor
from methods.lossless.zfp import Zfp
import time
import main


def test_compressor(output_dir: str, output_compressed_file: str, compressor: Compressor, similarity: bool,
                    holo_file_name: str, hologram: HoloSpec) -> None:
    if not os.path.isdir(os.path.join(output_dir, holo_file_name)):
        os.makedirs(os.path.join(output_dir, holo_file_name))

    output_path = os.path.join(output_dir, holo_file_name, output_compressed_file)
    start_time = time.time()
    compressor.compress(hologram, output_path)
    end_time = time.time()
    print(f"Tempo compressione: {end_time - start_time} secondi")

    ratio = compressor.calculate_info(output_path, holo_file_name)
    print(f"Ratio {ratio}")

    compressed_len = os.path.getsize(output_path)
    uncompressed_len = os.path.getsize(holo_file_name)
    HoloUtils.rate(uncompressed_len, compressed_len)

    if similarity:
        start_time = time.time()
        new_holo = compressor.decompress(output_path)
        end_time = time.time()
        print(f"Tempo decompressione: {end_time - start_time} secondi")
        similarity = paper_similarity.Similarity(paper_similarity.GammaM.bump, paper_similarity.GammaR.cos,
                                                 paper_similarity.GammaA.unique)
        print(f"Similarity: {similarity.calc_similarity(hologram.holo, new_holo.holo)}")
    print()


def main3():
    holo_file_name = 'mat_files/Hol_2D_dice.mat'
    hologram = main.open_hologram(holo_file_name)

    # BZIP2
    print("BZIP2")
    test_compressor('bzip2_compression', 'bzip2_compressed.bin', Bzip2Compressor(), True,
                    holo_file_name, hologram)

    # BZIP2GER
    print("BZIP2GER")
    test_compressor('bzip2ger_compression', 'bzip2ger_compressed.bin', Bzip2GerCompressor(), False,
                    holo_file_name, hologram)

    # GZIP
    print("GZIP")
    test_compressor('gzip_compression', 'gzip_compressed.bin', GzipCompressor(), True,
                    holo_file_name, hologram)

    # GZIPGER
    print("GZIPGER")
    test_compressor('gzipger_compression', 'gzipger_compressed.bin', GzipGerCompressor(), False,
                    holo_file_name, hologram)

    # ZIP
    print("ZIP")
    test_compressor('zip_compression', 'zip_compressed.bin', ZipCompressor(), True,
                    holo_file_name, hologram)

    # ZIPGER
    print("ZIPGER")
    test_compressor('zipger_compression', 'zipger_compressed.bin', ZipGerCompressor(), False,
                    holo_file_name, hologram)

    # LZMA
    print("LZMA")
    test_compressor('lzma_compression', 'lzma_compressed.bin', LZMACompressor(), True,
                    holo_file_name, hologram)

    # LZMA_GER
    print("LZMA_GER")
    test_compressor('lzmager_compression', 'lzmager_compressed.bin', LZMAGerCompressor(), False,
                    holo_file_name, hologram)

    # ZSTD
    print("ZSTD")
    test_compressor('zstd_compression', 'zstd_compressed.bin', ZSTDCompressor(), True,
                    holo_file_name, hologram)

    # ZSTD_GER
    print("ZSTD_GER")
    test_compressor('zstdger_compression', 'zstdger_compressed.bin', ZSTDGerCompressor(), False,
                    holo_file_name, hologram)

    # ZFP
    print("ZFP")
    test_compressor('zfp_compression', 'zfp_compressed.bin', Zfp(0), True,
                    holo_file_name, hologram)



if __name__ == '__main__':
    main3()


#%%
