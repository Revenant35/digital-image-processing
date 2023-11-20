import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def show_image(image: np.ndarray, name: str = "Image"):
    """
    Shows an image using OpenCV
    :param image:
    :param name:
    :return:
    """
    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main(image: np.ndarray, scale: int = 3):
    sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=scale)
    sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=scale)

    magnitude = cv2.magnitude(sobelx, sobely, None)
    orientation_radians = np.arctan2(sobely, sobelx)
    orientation = np.rad2deg(orientation_radians) % 360

    histogram, _ = np.histogram(orientation, bins=359, range=(1, 360))

    show_image(image, "Original")
    show_image(cv2.normalize(sobelx, None, 0.0, 255.0, cv2.NORM_MINMAX, cv.CV_8U), "Sobel X")
    show_image(cv2.normalize(sobely, None, 0.0, 255.0, cv2.NORM_MINMAX, cv.CV_8U), "Sobel Y")
    show_image(cv2.normalize(magnitude, None, 0.0, 255.0, cv2.NORM_MINMAX, cv.CV_8U), "Magnitude")
    show_image(cv2.normalize(orientation, None, 0.0, 255.0, cv2.NORM_MINMAX, cv.CV_8U), "Orientation")

    plt.plot(histogram)
    plt.show()


if __name__ == '__main__':
    image = cv.imread("../images/Naka1_small.tif")

    main(image, 3)
    # main(image, 11)


