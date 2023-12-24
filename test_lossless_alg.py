import os

from methods.lossless.general_algorithms import *
from methods.lossless.image_algorithms import *
from methods.general import paper_similarity
from methods.general.compressor import Compressor
from methods.general.compressor import HoloSpec
from methods.lossless.general_algorithms.general_compressor import GeneralCompressor
from methods.lossless.image_algorithms.image_compressor import ImageCompressor, PillowCompressor
from methods.lossless.zfpcompressor import ZfpCompressor
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

    ratio = compressor.calculate_info(output_path, holo_file_name)

    compressed_len = os.path.getsize(output_path)
    uncompressed_len = os.path.getsize(holo_file_name)
    # HoloUtils.rate(uncompressed_len, compressed_len)
    holo_original = uncompressed_len
    holo_compressed = compressed_len
    rate = (float(holo_compressed) / float(holo_original)) * 100
    print(output_compressed_file, "|", end_time - start_time, "|", ratio, "|", f"{(100 - rate):.2f}%", "|",
          str(int(holo_original / holo_compressed)) + ':1')

    if similarity:
        start_time = time.time()
        new_holo = compressor.decompress(output_path)
        end_time = time.time()
        print(f"Tempo decompressione: {end_time - start_time} secondi")
        similarity = paper_similarity.Similarity(paper_similarity.GammaM.bump, paper_similarity.GammaR.cos,
                                                 paper_similarity.GammaA.unique)
        print(f"Similarity: {similarity.calc_similarity(hologram.holo, new_holo.holo)}")
    print()


def generate_params():
    for floatizer in ["in_place", "hstack", "vstack"]:
        for g17 in [None, "bits", "bytes"]:
            yield floatizer, g17


def generate_all_image() -> (ImageCompressor, str, str | None):
    for algo in [PngCompressor, WebpCompressor]:
        for floatizer, g17 in generate_params():
            yield algo, floatizer, g17


def generate_all_general() -> (GeneralCompressor, str, str | None):
    for algo in [Bzip2Compressor, GzipCompressor, LzmaCompressor, ZipCompressor, ZstdCompressor]:
        for floatizer, g17 in generate_params():
            yield algo, floatizer, g17


def main3():
    holo_file_name = 'mat_files/Hol_2D_dice.mat'
    hologram = main.open_hologram(holo_file_name)
    # Algoritmo generici
    for algo, floatizer, g17 in generate_all_general():
        test_compressor("compressdir/" + algo.__name__, f"{algo.__name__}_{floatizer}_{str(g17)}.bin",
                        algo(floatizer, g17), False,
                        holo_file_name, hologram)
    # Image based
    for algo, floatizer, g17 in generate_all_image():
        if g17 == "bits":
            continue
        algor: PillowCompressor = algo(floatizer, g17)
        test_compressor("compressdir/" + algo.__name__, f"{algo.__name__}_{floatizer}_{str(g17)}.{algor.format_name}",
                        algor, False,
                        holo_file_name, hologram)
    # ZFP
    algo = ZfpCompressor(0)
    for floatizer, g17 in generate_params():
        test_compressor("compressdir/" + type(algo).__name__, f"{type(algo).__name__}_{floatizer}_{str(g17)}.bin", algo,
                        False,
                        holo_file_name, hologram)


if __name__ == '__main__':
    main3()

