import numpy as np
import sys
import logging
from file_operations import open_image, save_image, show_image, get_image_folder


def gray_to_binary(image, threshold):
    binary_image = np.zeros(image.shape, dtype=np.uint8)
    binary_image[image > threshold] = 255
    return binary_image


if __name__ == "__main__":
    image_folder = get_image_folder()

    if len(sys.argv) == 1:
        print("Please provide an input file")
        exit(1)

    if len(sys.argv) == 2:
        print("Please provide an output file")
        exit(1)

    if len(sys.argv) == 3 \
            or not sys.argv[3].isdigit() \
            or int(sys.argv[3]) < 0 \
            or int(sys.argv[3]) > 255:
        print("Please provide a valid threshold")
        exit(1)

    if len(sys.argv) > 4 and sys.argv[4] == "debug":
        logging.basicConfig(level=logging.DEBUG)

    INPUT_FILENAME = sys.argv[1]
    OUTPUT_FILENAME = sys.argv[2]
    THRESHOLD = int(sys.argv[3])

    INPUT_FILEPATH = image_folder.joinpath(INPUT_FILENAME)
    OUTPUT_FILEPATH = image_folder.joinpath(OUTPUT_FILENAME)

    if not INPUT_FILEPATH.exists():
        print("Input file does not exist")
        exit(1)

    image = open_image(str(INPUT_FILEPATH))

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        show_image(image, "input image")

    binary_image = gray_to_binary(image, THRESHOLD)

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        show_image(binary_image, "output image")

    save_image(binary_image, str(OUTPUT_FILEPATH))

    print("done")
