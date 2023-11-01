import numpy as np
import cv2
from file_operations import *
import matplotlib.pyplot as plt


def get_average_intensity(grayscale_image):
    avg = cv2.mean(grayscale_image)
    return round(avg[0])


def get_intensity_histogram(grayscale_image):
    hist = np.histogram(grayscale_image, bins=256, range=(0, 256))[0]
    return hist.astype(np.float64)


def get_normalized_intensity_histogram(grayscale_image):
    hist = get_intensity_histogram(grayscale_image)
    return np.divide(hist.ravel(), np.sum(hist))


def get_cumulative_normalized_intensity_histogram(grayscale_image):
    histogram = get_normalized_intensity_histogram(grayscale_image)
    return np.cumsum(histogram)


if __name__ == "__main__":

    image = open_RGB_image("river.jpg")
    assert image is not None, "Image 'river.jpg' not found"

    for channel in range(3):
        histogram = get_cumulative_normalized_intensity_histogram(image[:, :, channel])
        plt.plot(histogram)
        plt.xlim([0, 256])
        plt.show()
        average_intensity = get_average_intensity(image[:, :, channel])
        print(average_intensity)
