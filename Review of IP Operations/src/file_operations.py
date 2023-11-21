from pathlib import Path
import cv2 as cv
import numpy as np

IMAGE_DIRECTORY_PATH = Path(__file__).parents[1].joinpath('images')
assert IMAGE_DIRECTORY_PATH.exists(), f'Could not find {IMAGE_DIRECTORY_PATH}. Make sure the path is correct.'


def open_image(filename: str) -> np.ndarray:
    """
    Opens an image from the IMAGE_DIRECTORY_PATH

    Args:
        filename:  The name of the file to open

    Returns:
        A numpy array representing the image
    """
    filepath = IMAGE_DIRECTORY_PATH.joinpath(filename)
    image = cv.imread(str(filepath))
    return image


def show_image(image: np.ndarray, name: str = "Image") -> None:
    """
    Shows an image using OpenCV

    Args:
        image:  An image represented as a numpy array
        name:  The name of the window to display the image in

    Returns:
        None
    """
    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def save_image(image: np.ndarray, filename: str) -> None:
    """
    Saves an image to the IMAGE_DIRECTORY_PATH

    Args:
        image:  An image represented as a numpy array
        filename:  The name of the file to save the image to

    Returns:
        None
    """
    filepath = IMAGE_DIRECTORY_PATH.joinpath(filename)
    cv.imwrite(str(filepath), image)
