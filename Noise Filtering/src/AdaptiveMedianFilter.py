import time
import cv2 as cv
import numpy as np

from src.MSE import MSE
from src.util import *


def AdaptiveMedianFilter(image, initial_window_size, max_kernel_size):
    assert initial_window_size % 2 == 1
    assert max_kernel_size % 2 == 1
    assert initial_window_size <= max_kernel_size

    # Get the dimensions of the image
    H, W = image.shape

    # Create a copy of the image
    filtered_image = image.copy()

    # Add padding to the image
    pad_size = max_kernel_size // 2
    padded_image = cv.copyMakeBorder(
        image, pad_size, pad_size, pad_size, pad_size, cv.BORDER_CONSTANT, value=0
    )

    # Iterate over the image
    for y in range(H):
        for x in range(W):
            window_size = initial_window_size
            while window_size <= max_kernel_size:
                start_x = x + pad_size - window_size // 2
                start_y = y + pad_size - window_size // 2
                end_x = x + pad_size + window_size // 2 + 1
                end_y = y + pad_size + window_size // 2 + 1
                window = padded_image[start_y:end_y, start_x:end_x]

                z_min = np.min(window)
                z_max = np.max(window)
                z_med = np.median(window)
                z_xy = image[y, x]

                # Determine if z_med is an impulse
                if z_min < z_med < z_max:
                    # z_med is NOT an impulse; Determine if z_xy is an impulse
                    if z_min < z_xy < z_max:
                        # z_xy is NOT an impulse
                        filtered_image[y, x] = z_xy
                        break
                    else:
                        # z_xy is an impulse
                        filtered_image[y, x] = z_med
                        break
                else:
                    # z_med is an impulse; increase the window size
                    window_size += 2
                    if window_size > max_kernel_size:
                        # z_med is an impulse, but the window size is already at max_kernel_size
                        filtered_image[y, x] = z_med
                        break

    return filtered_image


if __name__ == "__main__":
    test1 = cv.imread("../images/Test1.png", cv.IMREAD_GRAYSCALE)
    test1_noisy = cv.imread("../images/Test1Noise2.png", cv.IMREAD_GRAYSCALE)
    test2 = cv.imread("../images/Test2.png", cv.IMREAD_GRAYSCALE)
    test2_noisy = cv.imread("../images/Test2Noise2.png", cv.IMREAD_GRAYSCALE)

    assert test1 is not None
    assert test1_noisy is not None
    assert test2 is not None
    assert test2_noisy is not None

    assert test1.shape == test1_noisy.shape
    assert test2.shape == test2_noisy.shape

    [test1_recovered_7x7, time_1_7x7] = time_function(
        AdaptiveMedianFilter, test1_noisy, 3, 7
    )
    [test1_recovered_19x19, time_1_19x19] = time_function(
        AdaptiveMedianFilter, test1_noisy, 3, 19
    )
    [test2_recovered_7x7, time_2_7x7] = time_function(
        AdaptiveMedianFilter, test2_noisy, 3, 7
    )
    [test2_recovered_19x19, time_2_19x19] = time_function(
        AdaptiveMedianFilter, test2_noisy, 3, 19
    )

    save_image(test1_recovered_7x7, "AMF_Test1Noise2_1.png")
    save_image(test1_recovered_19x19, "AMF_Test1Noise2_2.png")
    save_image(test2_recovered_7x7, "AMF_Test2Noise2_1.png")
    save_image(test2_recovered_19x19, "AMF_Test2Noise2_2.png")

    print("Test1Noise2")
    print("---------------------------------")
    print("MSE of Test1 vs Test1 (Noisy) is: ", MSE(test1, test1_noisy))
    print(
        "MSE of Test1 vs Test1 (Restored w/ AMF [3x3]->[7x7] : ",
        MSE(test1, test1_recovered_7x7),
    )
    print("time = ", time_1_7x7)
    print(
        "MSE of Test1 vs Test1 (Restored w/ AMF [3x3]->[19x19] : ",
        MSE(test1, test1_recovered_19x19),
    )
    print("time = ", time_1_19x19)
    print("---------------------------------")

    print("Test2Noise2")
    print("---------------------------------")
    print("MSE of Test2 vs Test2 (Noisy) is: ", MSE(test2, test2_noisy))
    print(
        "MSE of Test2 vs Test2 (Restored w/ AMF [3x3]->[7x7] : ",
        MSE(test2, test2_recovered_7x7),
    )
    print("time = ", time_2_7x7)
    print(
        "MSE of Test2 vs Test2 (Restored w/ AMF [3x3]->[19x19] : ",
        MSE(test2, test2_recovered_19x19),
    )
    print("time = ", time_2_19x19)
    print("---------------------------------")
