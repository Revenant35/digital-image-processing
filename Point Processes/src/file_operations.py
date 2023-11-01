from pathlib import Path
import cv2 as cv
import numpy as np


def rgb_to_gray(rgb_image):
    # Note: OpenCV uses BGR instead of RGB
    blue_image = (0.1140 * rgb_image[:, :, 0]).astype(np.uint8)
    green_image = (0.5871 * rgb_image[:, :, 1]).astype(np.uint8)
    red_image = (0.2989 * rgb_image[:, :, 2]).astype(np.uint8)
    return red_image + green_image + blue_image


def save_image(image, path):
    cv.imwrite(path, image)
    return


def open_image(path):
    return cv.imread(path)


def open_image_grayscale(path):
    image = cv.imread(path)

    # image is RGB, convert to grayscale
    if len(image.shape) == 3:
        image = rgb_to_gray(image)

    return image

def open_image_binary(path):
    image = cv.imread(path)

    # image is RGB, convert to grayscale
    if len(image.shape) == 3:
        image = rgb_to_gray(image)

    image = (image > 127).astype(np.uint8) * 255

    return image

def show_image(image, title):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_image_folder():
    folder = Path(__file__).parents[1].joinpath("images").resolve()
    assert folder.exists()
    return folder
