import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import logging
from file_operations import (
    show_image,
    get_image_folder,
    open_image_grayscale,
    open_image_binary,
    open_image,
    save_image,
)

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logging.disable(logging.DEBUG)

# perform a linear contrast stretch on the input image,
# discarding the specified percentage of pixels from either end of the histogram
def contrast_stretch(image, discard=0):
    # create an array of pixel values sorted from smallest to largest
    sorted_pixels = np.sort(image, axis=None)

    # grab the pixel indices at the low and high discard percentages
    low_index = int((discard / 100) * len(sorted_pixels))
    high_index = int((1 - (discard / 100)) * len(sorted_pixels))

    # grab the pixel values at the low and high indices
    min_pixel = sorted_pixels[low_index]
    max_pixel = sorted_pixels[high_index - 1]

    print("min pixel: ", sorted_pixels[low_index])
    print("max pixel: ", sorted_pixels[high_index - 1])

    # calculate the slope and y-intercept of the linear contrast stretch function
    coefficient = (255 / (max_pixel - min_pixel))
    y_intercept = -1 * coefficient * min_pixel

    # convert the image to a float64, so we can do math on it without overflowing
    image = image.astype(np.float64)

    # for each pixel in the input image, calculate the new pixel value
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # g(x) = a * f(x) + b
            image[i, j] = (coefficient * image[i, j]) + y_intercept
            if image[i, j] > 255:
                image[i, j] = 255
            elif image[i, j] < 0:
                image[i, j] = 0

    # convert the image back to uint8
    image = image.astype(np.uint8)

    # create an array of pixel values sorted from smallest to largest
    sorted_pixels = np.sort(image, axis=None)

    print("min pixel: ", sorted_pixels[0])
    print("max pixel: ", sorted_pixels[-1])

    return image


if __name__ == "__main__":
    image_folder = get_image_folder()

    if len(sys.argv) == 1:
        print("Please provide an input file")
        exit(1)

    INPUT_FILENAME = sys.argv[1]
    INPUT_FILEPATH = image_folder.joinpath(INPUT_FILENAME)

    if not INPUT_FILEPATH.exists():
        print("Input file does not exist")
        exit(1)

    image = open_image_grayscale(str(INPUT_FILEPATH))

    if len(image.shape) == 3:
        print("Please provide a grayscale image")
        exit(1)

    show_image(image, "input image")

    discard = 0

    if len(sys.argv) == 3:
        discard = float(sys.argv[2])
        if discard < 0 or discard > 100:
            print("Please provide a valid discard percentage [0-50)")
            exit(1)

    plt.hist(image.ravel(), bins=256, range=(0.0, 255.0), fc="k", ec="k")
    plt.show()

    new_image = contrast_stretch(image, discard)

    show_image(new_image, "output image")
    save_image(new_image, str(image_folder.joinpath("output.png")))

    plt.hist(new_image.ravel(), bins=256, range=(0.0, 255.0), fc="k", ec="k")
    plt.show()

    print("done")
