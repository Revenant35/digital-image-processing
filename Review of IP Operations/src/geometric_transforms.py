import cv2 as cv
import numpy as np


def show_image(image: np.ndarray, name: str = "Image"):
    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def translate(image, tx, ty):
    '''
    Translates an image by tx pixels in the x direction and ty pixels in the y direction
    :param image:
    :param tx:
    :param ty:
    :return shifted:
    '''
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    return shifted


def crop_scale(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, s: float = 1.0):
    '''
    Crops an image to the rectangle defined by (x1, y1) and (x2, y2) and then scales it by scale
    :param image:
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param s: scale factor (default 1.0)
    :return:
    '''
    assert 0 <= x1 < x2 <= image.shape[1], "x1 must be smaller than x2 and both must be within the image"
    assert 0 <= y1 < y2 <= image.shape[0], "y1 must be smaller than y2 and both must be within the image"
    cropped_image = image[y1:y2, x1:x2]
    scaled = cv.resize(cropped_image, None, fx=s, fy=s)
    return scaled


def vertical_flip(image: np.ndarray):
    '''
    Flips an image vertically
    :param image:
    :return:
    '''
    flipped = cv.flip(image, 0)
    return flipped


def horizontal_flip(image: np.ndarray):
    '''
    Flips an image horizontally
    :param image:
    :return:
    '''
    flipped = cv.flip(image, 1)
    return flipped


def rotate(image: np.ndarray, angle: int):
    '''
    Rotates an image by angle degrees
    :param image:
    :param angle:
    :return:
    '''
    assert -180 <= angle <= 180, "Angle must be between -180 and 180"
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1)
    rotated = cv.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return rotated


def fill(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, val: int):
    '''
    Fills a rectangle defined by (x1, y1) and (x2, y2) with val
    :param image:
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param val:
    :return:
    '''
    assert 0 <= x1 < x2 <= image.shape[1], "x1 must be smaller than x2 and both must be within the image"
    assert 0 <= y1 < y2 <= image.shape[0], "y1 must be smaller than y2 and both must be within the image"
    assert 0 <= val <= 255, "val must be between 0 and 255"
    filled = image.copy()
    filled[y1:y2, x1:x2] = val
    return filled


if __name__ == '__main__':
    image = cv.imread("../images/Naka1_small.tif")
    translated = translate(image, 500, 400)
    cropped = crop_scale(image, 500, 1, 1000, 800, 0.5)
    vflip = vertical_flip(image)
    hflip = horizontal_flip(image)
    rot = rotate(image, 30)
    filled = fill(image, 500, 1, 1000, 800, 150)
    show_image(image, "Original")
    show_image(translated, "Translated")
    show_image(cropped, "Cropped")
    show_image(vflip, "Vertical Flip")
    show_image(hflip, "Horizontal Flip")
    show_image(rot, "Rotated")
    show_image(filled, "Filled")

