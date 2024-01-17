import multiprocessing
import os.path
import pickle
from concurrent import futures

import matplotlib.pyplot as plt
import numpy as np

from methods.general import paper_similarity
from methods.general.compressor import Compressor, HoloSpec
from methods.lossless.fp_algorithms import FpzipCompressor
from methods.lossless.fp_algorithms.fp_algorithm import FpAlgorithm


def calc_mask(shape, center_row, cutoff_frequency, center_col, offset_x):
    rows, cols = shape
    mask = np.ones((rows, cols), dtype=np.complex128)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((offset_x + i - center_row) ** 2 + (j - center_col) ** 2)

            if distance > cutoff_frequency:
                mask[i, j] = 0
    return mask


def _apply_low_pass_filter_par(image_fourier, cutoff_frequency):
    rows, cols = image_fourier.shape
    center_row, center_col = rows // 2, cols // 2

    num_processes = multiprocessing.cpu_count()
    shapes = [x.shape for x in np.array_split(image_fourier, num_processes, axis=0)]
    offsets = [0] + [x[0] for x in shapes][:-1]
    for i in range(len(offsets) - 1):
        offsets[i + 1] = offsets[i] + offsets[i + 1]

    with futures.ProcessPoolExecutor() as executor:
        fut = [executor.submit(calc_mask, shape, center_row, cutoff_frequency, center_col, offset) for shape, offset in
               zip(shapes, offsets)]
        print(fut)
        results = []

        for future in fut:
            print(future)
            result_chunk = future.result()
            results.append(result_chunk)

    result_matrix = np.concatenate(results, axis=0)
    filtered_image = image_fourier * result_matrix

    return filtered_image


def _apply_low_pass_filter(image_fourier, cutoff_frequency):
    rows, cols = image_fourier.shape
    mask = np.ones((rows, cols), dtype=np.complex128)
    center_row, center_col = rows // 2, cols // 2
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            if distance > cutoff_frequency:
                mask[i, j] = 0

    filtered_image = image_fourier * mask
    return filtered_image


def _apply_high_pass_filter(image_fourier, threshold):
    rows, cols = image_fourier.shape
    mask = np.ones((rows, cols), dtype=np.complex128)
    center_row, center_col = rows // 2, cols // 2
    mask[center_row - threshold:center_row + threshold, center_col - threshold:center_col + threshold] = 0
    filtered_image = image_fourier * mask

    return filtered_image


class FourierCompressor(Compressor):
    def is_lossless(self) -> bool:
        return True

    def compress(self, hologram: HoloSpec, output_path: str) -> None:
        fourier_holo = np.fft.fftshift(np.fft.fft2(hologram.holo))
        filter_holo = self.pass_filter(fourier_holo, self.threshold)
        with open(output_path, "wb") as fp:
            a = (hologram.dist, hologram.wlen, hologram.pp,
                 self.float_compressor.compress_float_matrix(filter_holo))
            pickle.dump(a, fp)

    def decompress(self, input_path: str) -> HoloSpec:
        with open(input_path, 'rb') as f:
            dist, wlen, pp, holo = pickle.load(f)
        holo = self.float_compressor.decompress_float_matrix(holo)
        holo = np.fft.ifft2(np.fft.ifftshift(holo))
        return HoloSpec(holo, pp, wlen, dist)

    def __init__(self, threshold: int, float_compressor: FpAlgorithm, use_low_pass_filter: bool = True):
        super().__init__()
        self.pass_filter = _apply_low_pass_filter_par if use_low_pass_filter else _apply_high_pass_filter
        self.threshold = threshold
        self.float_compressor = float_compressor


xxx = HoloSpec.open_hologram('mat_files/Hol_2D_dice.mat')
similarity = paper_similarity.Similarity(paper_similarity.GammaM.bump, paper_similarity.GammaR.cos,
                                         paper_similarity.GammaA.unique)

array_sim = []
array_size = []

out_path = "/tmp/ramdisk/TODELETE"


def calc(threshold):
    l = FourierCompressor(threshold, FpzipCompressor("in_place", None))
    l.compress(xxx, out_path)
    new = l.decompress(out_path)
    sim = similarity.calc_similarity(new.holo, xxx.holo)
    size = os.path.getsize(out_path)
    with open("plots_output", "a") as fp:
        print(threshold, sim, size, sep=",", file=fp)
    array_sim.append(sim)
    array_size.append(size)


def plot(array, name, i):
    plt.plot(array)
    plt.legend([name])
    plt.savefig("plot")


def plot_all(i):
    plot(array_sim, "Similarity", i)
    plot(array_size, "Size", i)
    plot([x / y for x, y in zip(array_sim, array_size)], "Sim/Size", i)
    plot([x / y for x, y in zip(array_size, array_sim)], "Size/Sim", i)


for i in range(270, 281):
    calc(i)
    # if i > 0 and i % 50 == 0:
    #    plot_all(i)
