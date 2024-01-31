import numpy as np

from methods.general.compressor import Compressor, HoloSpec


def _reconst_matrix(U, sigma, V, start, end, jump):
    for i in range(start, end, jump):
        reconst_matrix = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
    return reconst_matrix


class OldSvd(Compressor):
    def __init__(self, k_value: int):
        self.k_value = k_value

    def compress(self, hologram: HoloSpec, output_path: str) -> None:
        holo = hologram.holo
        U_HOLO, SIGMA_HOLO, V_HOLO = np.linalg.svd(holo)

        U_HOLO_CUT = U_HOLO[:, :self.k_value]
        V_HOLO_CUT = V_HOLO[:self.k_value, :]
        SIGMA_HOLO_CUT = SIGMA_HOLO[:self.k_value]

        np.savez(output_path + 'U_HOLO_P', U_HOLO_CUT)
        np.savez(output_path + 'V_HOLO_P', V_HOLO_CUT)
        np.savez(output_path + 'SIGMA_HOLO_P', SIGMA_HOLO_CUT)

    def decompress(self, input_path: str) -> HoloSpec:
        with np.load(input_path + 'U_HOLO_P.npz') as data:
            U_COMPRESS = data['arr_0']
        with np.load(input_path + 'V_HOLO_P.npz') as data:
            V_COMPRESS = data['arr_0']
        with np.load(input_path + 'SIGMA_HOLO_P.npz') as data:
            SIGMA_COMPRESS = data['arr_0']

        matrix_rec = _reconst_matrix(U_COMPRESS, SIGMA_COMPRESS, V_COMPRESS, 5, self.k_value + 1, 5)

        return HoloSpec(matrix_rec, 0, 0, 0)

    def is_lossless(self) -> bool:
        return False
