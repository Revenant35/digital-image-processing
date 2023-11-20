import cv2
from pathlib import Path

IMAGE_DIRECTORY = Path(__file__).parents[1] / "images"
assert IMAGE_DIRECTORY.exists(), f"Path {IMAGE_DIRECTORY} does not exist."


def read_image(filename):
    filepath = IMAGE_DIRECTORY / filename
    assert filepath.exists(), f"Path {filepath} does not exist."
    return cv2.imread(str(filepath))


def save_image(image, filename):
    filepath = IMAGE_DIRECTORY / filename
    cv2.imwrite(str(filepath), image)


def show_image(image, title="image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
