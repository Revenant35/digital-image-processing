import time

import cv2 as cv
from pathlib import Path

IMAGE_DIRECTORY = Path(__file__).parents[1].joinpath("images")


def open_image(filename):
    filepath = IMAGE_DIRECTORY.joinpath(filename)
    image = cv.imread(str(filepath), cv.IMREAD_GRAYSCALE)
    return image


def save_image(image, filename):
    filepath = IMAGE_DIRECTORY.joinpath(filename)
    cv.imwrite(str(filepath), image)


def show_image(image, title):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def time_function(function, *args):
    t = time.perf_counter()
    value = function(*args)
    return [value, time.perf_counter() - t]
