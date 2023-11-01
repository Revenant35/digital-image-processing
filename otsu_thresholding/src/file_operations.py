import cv2
from pathlib import Path

from matplotlib import pyplot as plt

images_folder = Path(__file__).parents[1].joinpath("images")
assert images_folder.exists(), "Images folder not found"


def open_RGB_image(filename):
    path = images_folder.joinpath(filename)
    if not path.exists() or not path.is_file():
        print(f"file: {filename} not found")
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def open_grayscale_image(filename):
    path = images_folder.joinpath(filename)
    if not path.exists() or not path.is_file():
        print(f"file: {filename} not found")
        return None
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def save_image(filename, image):
    path = images_folder.joinpath(filename)
    cv2.imwrite(str(path), image)


def save_figure(filename):
    path = images_folder.joinpath(filename)
    plt.savefig(str(path))


def show_image_with_title(image, title="Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image = open_RGB_image("river.jpg")
    assert image is not None, "Image 'river.jpg' not found"

    show_image_with_title(image, "River")

    image = open_grayscale_image("lungs.jpg")
    assert image is not None, "Image 'lungs.jpg' not found"

    show_image_with_title(image, "Lungs")

    image = open_grayscale_image("square.jpg")
    assert image is not None, "Image 'square.jpg' not found"

    show_image_with_title(image, "Square")
