import csv
import os
import timeit

from methods.general import paper_similarity
from methods.general.compressor import Compressor
from methods.general.compressor import HoloSpec
from methods.lossless.fp_algorithms import *
from methods.lossless.general_algorithms import *
from methods.lossless.general_algorithms.general_compressor import GeneralCompressor
from methods.lossless.image_algorithms import *
from methods.lossless.image_algorithms.image_compressor import ImageCompressor, PillowCompressor


def calculate_info(compressed_path: str, uncompressed_path: str) -> float:
    compressed_len = os.path.getsize(compressed_path)
    uncompressed_len = os.path.getsize(uncompressed_path)
    ratio = compressed_len / uncompressed_len
    return ratio


def test_compressor(csv_writer, output_dir: str, output_compressed_file: str, compressor: Compressor,
                    similarity: paper_similarity.Similarity | None,
                    holo_file_name: str, hologram: HoloSpec) -> None:
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, output_compressed_file)
    compress_time = timeit.timeit(lambda: compressor.compress(hologram, output_path), number=3)
    decompress_time = timeit.timeit(lambda: compressor.decompress(output_path), number=3)
    ratio = calculate_info(output_path, holo_file_name)
    data = [output_compressed_file, compress_time, decompress_time, ratio]
    if similarity is not None:
        new_holo = compressor.decompress(output_path)
        if compressor.is_lossless():
            if new_holo != hologram:
                raise RuntimeError(output_compressed_file + " is not lossless")
            data.append(1.0)
        else:
            simil = similarity.calc_similarity(hologram.holo, new_holo.holo)
            data.append(simil)
    if csv_writer is not None:
        csv_writer.writerow(data)
    print(data)


def generate_params():
    for floatizer in ["in_place", "hstack", "vstack"]:
        for g17 in [None, "bits", "bytes"]:
            yield floatizer, g17


def generate_all_image() -> (ImageCompressor, str, str | None):
    for algo in [PngCompressor, WebpCompressor]:
        for floatizer, g17 in generate_params():
            yield algo, floatizer, g17


def generate_all_fp() -> (FpAlgorithm, str, str | None):
    for algo in [FpzipCompressor, ZfpCompressor]:
        for floatizer, g17 in generate_params():
            if g17 == "bits":
                continue
            yield algo, floatizer, g17


def generate_all_general() -> (GeneralCompressor, str, str | None):
    for algo in [Bzip2Compressor, GzipCompressor, LzmaCompressor, ZipCompressor, ZstdCompressor]:
        for floatizer, g17 in generate_params():
            yield algo, floatizer, g17


def calc_generic(csv_writer, similarity: paper_similarity.Similarity, hologram: HoloSpec, holo_file_name: str):
    for algo, floatizer, g17 in generate_all_general():
        test_compressor(csv_writer, os.path.join("compressdir", algo.__name__),
                        f"{algo.__name__}_{floatizer}_{str(g17)}.bin",
                        algo(floatizer, g17), similarity, holo_file_name, hologram)


def calc_images(csv_writer, similarity: paper_similarity.Similarity, hologram: HoloSpec, holo_file_name: str):
    for algo, floatizer, g17 in generate_all_image():
        algor: PillowCompressor = algo(floatizer, g17)
        test_compressor(csv_writer, os.path.join("compressdir", algo.__name__),
                        f"{algo.__name__}_{floatizer}_{str(g17)}.{algor.format_name}",
                        algor, similarity, holo_file_name, hologram)


def calc_fp(csv_writer, similarity: paper_similarity.Similarity, hologram: HoloSpec, holo_file_name: str):
    for algo, floatizer, g17 in generate_all_fp():
        algo = algo(floatizer, g17)
        test_compressor(csv_writer, os.path.join("compressdir", type(algo).__name__),
                        f"{type(algo).__name__}_{floatizer}_{str(g17)}.bin",
                        algo, similarity, holo_file_name, hologram)


def main3():
    holo_file_name = 'mat_files/Hol_2D_dice.mat'
    similarity = paper_similarity.Similarity(paper_similarity.GammaM.bump, paper_similarity.GammaR.cos,
                                             paper_similarity.GammaA.unique)
    hologram = HoloSpec.open_hologram(holo_file_name)
    with open("benchmark.csv", "w") as fp:
        csv_writer = csv.writer(fp)
        headers = ["name", "compress_time", "decompress_time", "ratio"]
        if similarity is not None:
            headers.append("similarity")
        csv_writer.writerow(headers)
        calc_generic(csv_writer, similarity, hologram, holo_file_name)
        calc_images(csv_writer, similarity, hologram, holo_file_name)
        calc_fp(csv_writer, similarity, hologram, holo_file_name)


if __name__ == '__main__':
    main3()
