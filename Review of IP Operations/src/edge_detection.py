import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from file_operations import open_image, save_image


def main(image: np.ndarray, scale: int = 3):
    sobel_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=scale)
    sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=scale)

    magnitude = cv2.magnitude(sobel_x, sobel_y, None)
    orientation_radians = np.arctan2(sobel_y, sobel_x)
    orientation = np.rad2deg(orientation_radians) % 360

    histogram, _ = np.histogram(orientation, bins=359, range=(1, 360))

    normalized_sobel_x = cv2.normalize(sobel_x, None, 0.0, 255.0, cv2.NORM_MINMAX, cv.CV_8U)
    normalized_sobel_y = cv2.normalize(sobel_y, None, 0.0, 255.0, cv2.NORM_MINMAX, cv.CV_8U)
    normalized_magnitude = cv2.normalize(magnitude, None, 0.0, 255.0, cv2.NORM_MINMAX, cv.CV_8U)
    normalized_orientation = cv2.normalize(orientation, None, 0.0, 255.0, cv2.NORM_MINMAX, cv.CV_8U)

    save_image(normalized_sobel_x, f"sobel_x_{scale}.png")
    save_image(normalized_sobel_y, f"sobel_y_{scale}.png")
    save_image(normalized_magnitude, f"magnitude_{scale}.png")
    save_image(normalized_orientation, f"orientation_{scale}.png")

    plt.title(f"Orientation Histogram (k={scale})")
    plt.plot(histogram)
    plt.show()


if __name__ == '__main__':
    original_image = open_image("Naka1_small.tif")

    main(original_image, 3)
    main(original_image, 11)
