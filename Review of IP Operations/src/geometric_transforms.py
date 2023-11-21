import cv2 as cv
import numpy as np
from file_operations import open_image, save_image


def translate(image: np.ndarray, tx: int, ty: int) -> np.ndarray:
    """
    Translates an image by tx pixels in the x direction and ty pixels in the y direction

    Args:
        image:  A numpy array representing an image
        tx:  The number of pixels to translate the image in the x direction
        ty:  The number of pixels to translate the image in the y direction
        
    Returns:
        A numpy array representing the translated image
    """
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    return shifted


def crop_scale(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, s: float = 1.0) -> np.ndarray:
    """
    Crops an image to the rectangle defined by (x1, y1) and (x2, y2) and then scales it by scale

    Args:
        image:  A numpy array representing an image
        x1:  The x coordinate of the top left corner of the rectangle to crop
        y1:  The y coordinate of the top left corner of the rectangle to crop
        x2:  The x coordinate of the bottom right corner of the rectangle to crop
        y2:  The y coordinate of the bottom right corner of the rectangle to crop
        s:  The scale to resize the image to
        
    Returns:
        A numpy array representing the cropped and scaled image
    """
    assert 0 <= x1 < x2 <= image.shape[1], "x1 must be smaller than x2 and both must be within the image"
    assert 0 <= y1 < y2 <= image.shape[0], "y1 must be smaller than y2 and both must be within the image"
    cropped = image[y1:y2, x1:x2]
    scaled = cv.resize(cropped, None, fx=s, fy=s)
    return scaled


def vertical_flip(image: np.ndarray) -> np.ndarray:
    """
    Flips an image vertically

    Args:
        image:  A numpy array representing an image

    Returns:
        A numpy array representing the vertically flipped image
    """
    flipped = cv.flip(image, 0)
    return flipped


def horizontal_flip(image: np.ndarray) -> np.ndarray:
    """
    Flips an image horizontally

    Args:
        image:  A numpy array representing an image

    Returns:
        A numpy array representing the horizontally flipped image
    """
    flipped = cv.flip(image, 1)
    return flipped


def rotate(image: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotates an image by angle degrees

    Args:
        image:  A numpy array representing an image
        angle:  The angle to rotate the image by

    Returns:
        A numpy array representing the rotated image
    """
    assert -180 <= angle <= 180, "Angle must be between -180 and 180"
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1)
    rotated = cv.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return rotated


def fill(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, val: int) -> np.ndarray:
    """
    Fills the rectangle defined by (x1, y1) and (x2, y2) with the value val

    Args:
        image:  A numpy array representing an image
        x1:  The x coordinate of the top left corner of the rectangle to fill
        y1:  The y coordinate of the top left corner of the rectangle to fill
        x2:  The x coordinate of the bottom right corner of the rectangle to fill
        y2:  The y coordinate of the bottom right corner of the rectangle to fill
        val:  The value to fill the rectangle with

    Returns:
        A numpy array representing the filled image
    """
    assert 0 <= x1 < x2 <= image.shape[1], "x1 must be smaller than x2 and both must be within the image"
    assert 0 <= y1 < y2 <= image.shape[0], "y1 must be smaller than y2 and both must be within the image"
    assert 0 <= val <= 255, "val must be between 0 and 255"
    filled = image.copy()
    filled[y1:y2, x1:x2] = val
    return filled


if __name__ == '__main__':
    original_image = open_image("Naka1_small.tif")

    translated = translate(original_image, 500, 400)
    cropped_image = crop_scale(original_image, 500, 1, 1000, 800, 0.5)
    vertically_flipped_image = vertical_flip(original_image)
    horizontally_flipped_image = horizontal_flip(original_image)
    rotated_image = rotate(original_image, 30)
    filled_image = fill(original_image, 500, 1, 1000, 800, 150)

    # Save the images
    save_image(original_image, "Naka1_small.png")
    save_image(translated, "Naka1_small_translated.png")
    save_image(cropped_image, "Naka1_small_cropped.png")
    save_image(vertically_flipped_image, "Naka1_small_vertically_flipped.png")
    save_image(horizontally_flipped_image, "Naka1_small_horizontally_flipped.png")
    save_image(rotated_image, "Naka1_small_rotated.png")
    save_image(filled_image, "Naka1_small_filled.png")
