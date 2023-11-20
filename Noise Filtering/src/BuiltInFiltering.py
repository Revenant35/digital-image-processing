import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.MSE import MSE
from src.util import *


def GaussianFilter(image, sigma):
    blurred_image = cv.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return blurred_image


def MedianFilter(image, kernel_size):
    blurred_image = cv.medianBlur(image, kernel_size)
    return blurred_image


if __name__ == "__main__":
    test1 = cv.imread("../images/Test1.png", cv.IMREAD_GRAYSCALE)
    test1_noise1 = cv.imread("../images/Test1Noise1.png", cv.IMREAD_GRAYSCALE)
    test1_noise2 = cv.imread("../images/Test1Noise2.png", cv.IMREAD_GRAYSCALE)

    assert test1 is not None
    assert test1_noise1 is not None
    assert test1_noise2 is not None

    assert test1.shape == test1_noise1.shape == test1_noise2.shape

    test1_noise1_restored_sigma2 = GaussianFilter(test1_noise1, 2)
    test1_noise1_restored_sigma7 = GaussianFilter(test1_noise1, 7)
    test1_noise2_restored_sigma2 = GaussianFilter(test1_noise2, 2)
    test1_noise2_restored_sigma7 = GaussianFilter(test1_noise2, 7)

    test1_noise1_restored_kernel7 = MedianFilter(test1_noise1, 7)
    test1_noise1_restored_kernel19 = MedianFilter(test1_noise1, 19)
    test1_noise2_restored_kernel7 = MedianFilter(test1_noise2, 7)
    test1_noise2_restored_kernel19 = MedianFilter(test1_noise2, 19)

    save_image(test1_noise1_restored_sigma2, "Gaussian_noise1_sigma2.png")
    save_image(test1_noise1_restored_sigma7, "Gaussian_noise1_sigma7.png")
    save_image(test1_noise2_restored_sigma2, "Gaussian_noise2_sigma2.png")
    save_image(test1_noise2_restored_sigma7, "Gaussian_noise2_sigma7.png")

    save_image(test1_noise1_restored_kernel7, "Median_noise1_kernel7.png")
    save_image(
        test1_noise1_restored_kernel19,
        "Median_noise1_kernel19.png",
    )
    save_image(test1_noise2_restored_kernel7, "Median_noise2_kernel7.png")
    save_image(
        test1_noise2_restored_kernel19,
        "Median_noise2_kernel19.png",
    )

    print("Test1Noise1")
    print("---------------------------------")
    print("MSE of Ground Truth vs Noise1 is: ", MSE(test1, test1_noise1))
    print(
        "MSE of Ground Truth vs Noise1 w/ Gaussian (Sigma = 2): ",
        MSE(test1, test1_noise1_restored_sigma2),
    )
    print(
        "MSE of Ground Truth vs Noise1 w/ Gaussian (Sigma = 7): ",
        MSE(test1, test1_noise1_restored_sigma7),
    )
    print(
        "MSE of Ground Truth vs Noise1 w/ Median (Kernel = 7): ",
        MSE(test1, test1_noise1_restored_kernel7),
    )
    print(
        "MSE of Ground Truth vs Noise1 w/ Median (Kernel = 19): ",
        MSE(test1, test1_noise1_restored_kernel19),
    )
    print("---------------------------------")

    print("Test1Noise2")
    print("---------------------------------")
    print("MSE of Ground Truth vs Noise2 is: ", MSE(test1, test1_noise2))
    print(
        "MSE of Ground Truth vs Noise2 w/ Gaussian (Sigma = 2): ",
        MSE(test1, test1_noise2_restored_sigma2),
    )
    print(
        "MSE of Ground Truth vs Noise2 w/ Gaussian (Sigma = 7): ",
        MSE(test1, test1_noise2_restored_sigma7),
    )
    print(
        "MSE of Ground Truth vs Noise2 w/ Median (Kernel = 7): ",
        MSE(test1, test1_noise2_restored_kernel7),
    )
    print(
        "MSE of Ground Truth vs Noise2 w/ Median (Kernel = 19): ",
        MSE(test1, test1_noise2_restored_kernel19),
    )
    print("---------------------------------")
