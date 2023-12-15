import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import main


def render_val(val, output_file):
    x = np.unpackbits(np.array([val]).view(np.uint8)) * 255
    x = x.reshape((-1, 8))
    cv2.imwrite(output_file, x, params=[cv2.IMWRITE_PNG_COMPRESSION, 9])


def process_frame(args):
    idx, val = args
    render_val(val, f"frames2/frame{idx + 10:010d}.png")


def main2():
    holoFileName = 'mat_files/Hol_2D_dice.mat'
    orig = main.open_hologram(holoFileName)
    render_val(np.uint64(orig.holo.shape[0]).astype(np.dtype("<U16")), "frames2/frame0000000000.png")
    render_val(np.uint64(orig.holo.shape[1]).astype(np.dtype("<U16")), "frames2/frame0000000001.png")
    # render_val(np.complex128(orig.pp  ).view(np.dtype("<U16")), "frames2/frame0000000002.png")
    # render_val(np.complex128(orig.wlen).view(np.dtype("<U16")), "frames2/frame0000000003.png")
    # render_val(np.complex128(orig.dist).view(np.dtype("<U16")), "frames2/frame0000000004.png")

    with ProcessPoolExecutor() as executor:
        # Enumerate over flattened array and submit tasks to the process pool
        futures = executor.map(process_frame, enumerate(orig.holo.flatten(order="C")))

    # Wait for all tasks to complete
    for future in futures:
        pass  # Accessing the result of the future will raise an exception if the function call raised one


if __name__ == '__main__':
    main2()
