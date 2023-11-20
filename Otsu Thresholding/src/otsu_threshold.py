import time

import numpy as np
import cv2
from intensity_histogram import *
from file_operations import *


# Note: q is the cumulative sum of the normalized intensity histogram,
#       so I just calculate this faster using numpy.cumsum()
def recursive_otsu_threshold(image):
    hist = get_normalized_intensity_histogram(image)
    q = np.cumsum(hist)
    average_intensity = get_average_intensity(image)

    mu_1 = np.zeros(256)
    mu_2 = np.zeros(256)

    last_zero_index = 0
    while q[last_zero_index + 1] == 0:
        last_zero_index += 1

    last_one_index = 255
    while q[last_one_index] == 1:
        last_one_index -= 1

    for i in range(last_zero_index, last_one_index):
        mu_1[i + 1] = ((q[i] * mu_1[i]) + ((i + 1) * hist[i + 1])) / q[i + 1]
        mu_2[i + 1] = (average_intensity - (q[i + 1] * mu_1[i + 1])) / (1 - q[i + 1])
        if mu_1[i + 1] < 0:
            mu_1[i + 1] = 0
        if mu_2[i + 1] < 0:
            mu_2[i + 1] = 0
        if mu_1[i + 1] > 255:
            mu_1[i + 1] = 255
        if mu_2[i + 1] > 255:
            mu_2[i + 1] = 255

    max = -np.inf
    threshold = -1

    for i in range(0, 255):
        sigma = q[i] * (1 - q[i]) * ((mu_1[i] - mu_2[i]) ** 2)
        if sigma > max:
            max = sigma
            threshold = i

    return threshold


def binarize(image, threshold):
    return np.where(image > threshold, 255, 0).astype(np.uint8)


def run_river():
    image = open_RGB_image("river.jpg")
    for i in range(3):
        color = ["blue", "green", "red"][i]
        show_image_with_title(image[:, :, i], "River")
        save_image(f"river_{color}.jpg", image[:, :, i])

        hist = get_intensity_histogram(image[:, :, i])
        plt.plot(hist)
        plt.xlim([0, 256])
        save_figure(f"river_{color}_histogram.jpg")
        plt.clf()

        start = time.perf_counter()
        threshold = recursive_otsu_threshold(image[:, :, i])
        end = time.perf_counter()
        binarized_image = binarize(image[:, :, i], threshold)

        print(f"River {color} Results:")
        print(f"Time: {end - start}")
        print(f"Threshold: {round(threshold)}\n")

        show_image_with_title(binarized_image, f"Binarized {color}")
        save_image(f"river_binarized_{color}.jpg", binarized_image)


def run_lungs():
    image = open_grayscale_image("lungs.jpg")
    show_image_with_title(image, "Lungs")

    hist = get_intensity_histogram(image)
    plt.plot(hist)
    plt.xlim([0, 256])
    save_figure(f"lungs_histogram.jpg")
    plt.clf()

    start = time.perf_counter()
    threshold = recursive_otsu_threshold(image)
    end = time.perf_counter()

    print("Lungs Results:")
    print(f"Time: {end - start}")
    print(f"Threshold: {round(threshold)}\n")

    binarized_image = binarize(image, threshold)
    show_image_with_title(binarized_image, "Binarized")
    save_image("lungs_binarized.jpg", binarized_image)

    inverted_binarized_image = cv2.bitwise_not(binarized_image)
    show_image_with_title(inverted_binarized_image, "Inverted Binarized")
    save_image("lungs_binarized_inverted.jpg", inverted_binarized_image)


def run_square():
    image = open_grayscale_image("square.jpg")
    show_image_with_title(image, "Square")

    hist = get_intensity_histogram(image)
    plt.plot(hist)
    plt.xlim([0, 256])
    save_figure(f"square_histogram.jpg")
    plt.clf()

    blurred_image = cv2.GaussianBlur(image, (495, 495), sigmaX=np.inf, sigmaY=np.inf)
    blurred_image = cv2.normalize(blurred_image, None, 0, 255, cv2.NORM_MINMAX)
    gradient = cv2.subtract(blurred_image, image)
    gradient = gradient / np.max(gradient)
    gradient = (gradient * 255).astype(np.uint8)

    image_without_gradient = cv2.subtract(image, gradient)
    show_image_with_title(image_without_gradient, "Square without gradient")
    save_image("square_without_gradient.jpg", image_without_gradient)

    hist = get_intensity_histogram(image_without_gradient)
    plt.plot(hist)
    plt.xlim([0, 256])
    save_figure(f"square_pps_histogram.jpg")
    plt.clf()

    start = time.perf_counter()
    threshold = recursive_otsu_threshold(image_without_gradient)
    end = time.perf_counter()

    print("Square Results:")
    print(f"Time: {end - start}")
    print(f"Threshold: {round(threshold)}\n")

    binarized_image = binarize(image_without_gradient, threshold)
    show_image_with_title(binarized_image, "Binarized")
    save_image("square_binarized.jpg", binarized_image)


if __name__ == "__main__":
    run_river()
    run_lungs()
    run_square()
