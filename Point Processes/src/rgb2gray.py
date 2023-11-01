import numpy as np
import sys
import logging
from file_operations import open_image, save_image, show_image, get_image_folder


def rgb_to_gray(rgb_image):
    # Note: OpenCV uses BGR instead of RGB
    blue_image = (0.1140 * rgb_image[:, :, 0]).astype(np.uint8)
    green_image = (0.5871 * rgb_image[:, :, 1]).astype(np.uint8)
    red_image = (0.2989 * rgb_image[:, :, 2]).astype(np.uint8)
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        show_image(blue_image, "blue")
        show_image(green_image, "green")
        show_image(red_image, "red")
    return red_image + green_image + blue_image


if __name__ == "__main__":
    image_folder = get_image_folder()

    if len(sys.argv) == 1:
        print("Please provide an input file")
        exit(1)

    if len(sys.argv) == 2:
        print("Please provide an output file")
        exit(1)

    if len(sys.argv) > 3 and sys.argv[3] == "debug":
        logging.basicConfig(level=logging.DEBUG)

    INPUT_FILENAME = sys.argv[1]
    OUTPUT_FILENAME = sys.argv[2]

    INPUT_FILEPATH = image_folder.joinpath(INPUT_FILENAME)
    OUTPUT_FILEPATH = image_folder.joinpath(OUTPUT_FILENAME)

    if not INPUT_FILEPATH.exists():
        print("Input file does not exist")
        exit(1)

    image = open_image(str(INPUT_FILEPATH))

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        show_image(image, "input image")

    gray_image = rgb_to_gray(image)

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        show_image(gray_image, "gray")

    save_image(gray_image, str(OUTPUT_FILEPATH))
