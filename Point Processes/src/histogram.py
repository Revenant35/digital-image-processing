import time
import matplotlib.pyplot as plt
import numpy as np
import sys
import logging
from file_operations import (
    open_image,
    show_image,
    get_image_folder,
    open_image_binary,
    save_image,
)


def myimhist(image, nbins=256, mask=None):
    intensity_occurances = np.zeros(256, dtype=np.uint64)
    if mask is not None:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if mask[i, j] == 255:
                    intensity_occurances[image[i, j]] += 1
    else:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                intensity_occurances[image[i, j]] += 1

    bin_size = 256 / nbins
    hist = np.zeros(nbins, dtype=np.uint64)
    for index, value in enumerate(intensity_occurances):
        bin_index = int(index / bin_size)
        hist[bin_index] += value

    return hist


if __name__ == "__main__":
    image_folder = get_image_folder()

    if len(sys.argv) == 1:
        print("Please provide an input file")
        print('Example: python histogram.py [input_file] [nbits] [mask_file (optional)]')
        exit(1)

    INPUT_FILENAME = sys.argv[1]
    INPUT_FILEPATH = image_folder.joinpath(INPUT_FILENAME)

    if not INPUT_FILEPATH.exists():
        print("Input file does not exist")
        exit(1)

    image = open_image(str(INPUT_FILEPATH))

    if len(sys.argv) == 2:
        print("Please provide a number of bins")
        print('Example: python histogram.py [input_file] [nbits] [mask_file (optional)]')
        exit(1)

    nbins = int(sys.argv[2])

    if nbins < 0 or nbins > 256:
        print("Please provide a valid number of bins")
        print('Range: 0 - 256')
        exit(1)

    mask = None

    if len(sys.argv) == 4:
        MASK_FILENAME = sys.argv[3]
        MASK_FILEPATH = image_folder.joinpath(MASK_FILENAME)

        if not MASK_FILEPATH.exists():
            print("Mask file does not exist")
            exit(1)

        mask = open_image_binary(str(MASK_FILEPATH))

    show_image(image, "input image")

    if mask is not None:
        print("mask provided")
        show_image(mask, "input mask")
    else:
        print("mask not provided")

    # iterate through each channel of image
    colors = ["blue", "green", "red"]
    for i in range(image.shape[2]):
        channel = image[..., i]
        # show_image(channel, f"{colors[i]} channel")
        save_image(channel, str(image_folder.joinpath(f"{colors[i]}_channel.png")))
        start = time.perf_counter()
        hist = myimhist(channel, nbins, mask)
        end = time.perf_counter()
        print(f"myimhist for {colors[i]} channel took {end - start:0.4f} seconds")
        plt.bar(np.arange(nbins), hist)
        plt.show()

    print("done")
