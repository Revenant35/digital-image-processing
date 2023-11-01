import cv2
from file_operations import read_image, save_image
import numpy as np
import scipy.ndimage as ndi
from src.preprocessing import preprocess_image


def scaled_convolution(image, log, sigma, alpha=1.0, positive=True):
    """
    Apply a log scaled convolution to an image.
    Based on: https://ieeexplore.ieee.org/abstract/document/6408211
    R_i = N(I * (-K_i))
    , use positive image
    For , use negative image (255 - image)

    Args:
    - image (numpy.ndarray): The input image.
    - log_filter (numpy.ndarray): The log filter.
    - sigma (float): Standard deviation of the Gaussian.
    - alpha (float): Blob-center detection accuracy
    - positive (bool): For bright blobs w/ dark background, enable (default). For dark blobs w/ bright background, disable.

    Returns:
    - numpy.ndarray: The image filtered by the log filter.
    """
    if not positive:
        image = 255 - image

    convolved = cv2.filter2D(image, -1, -log)  # I * (-K_i)
    scaled = (1 + np.log(sigma) ** alpha) ** 2 * convolved  # N(I * (-K_i))
    return scaled


def laplacian_of_gaussian(size, sigma):
    """
    Create a Laplacian of Gaussian (LoG) filter.

    Args:
    - size (int): The size of the filter. (should be odd)
    - sigma (float): Standard deviation of the Gaussian.

    Returns:
    - numpy.ndarray: The LoG filter.
    """
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    x, y = np.meshgrid(ax, ax)

    h = -((x ** 2) + (y ** 2)) / (2 * sigma ** 2)
    d = (-1 / (np.pi * sigma ** 4)) * (1 - (x ** 2 + y ** 2) / (2 * sigma ** 2))

    kernel = d * np.exp(h)

    return kernel


def blob_detection(image, sigma=4.0, threshold=0, filename=""):
    """
    Create a Laplacian of Gaussian (LoG) filter.
    Based on: https://ieeexplore.ieee.org/abstract/document/6408211

    Args:
    - image (numpy.ndarray): The input image.
    - sigma (float): Standard deviation of the Gaussian. Default is 4.
    - threshold (float): The minimum intensity of a blob. Default is 0.
    - filename (str): The filename of the image. Default is "".

    Returns:
    - None
    """

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_image(image, f"{filename}_grayscale.jpg")

    # blob radius is roughly 3 * sigma + 1
    size = int(2 * np.ceil(3 * sigma + 1) + 1)

    # Apply LoG filter
    log = laplacian_of_gaussian(size, sigma)

    # Scale the filtered image
    scaled = scaled_convolution(image, log, sigma, 1, False)

    # Find local maxima
    local_max = ndi.maximum_filter(scaled, size=size // 2 + 1, mode='nearest')

    # calculate the mask of maximums that are equal to the local maxima and greater than the threshold
    mask = (scaled == local_max) & (local_max > threshold)
    save_image(mask.astype(np.uint8) * 255, f"{filename}_mask.jpg")

    # Normalize the image to [0, 255]
    scaled /= np.max(scaled)
    scaled *= 255
    scaled = scaled.astype(np.uint8)
    save_image(scaled, f"{filename}_scaled_log.jpg")

    return mask


if __name__ == "__main__":
    image = read_image("GlomusTumor6.jpg")
    image = preprocess_image(image)
    save_image(image, f"GlomusTumor6_preprocessed.jpg")
    blob_detection(image, sigma=5, threshold=0, filename="GlomusTumor6")

    image = read_image("metastatic-breast-cancer.jpg")
    image = preprocess_image(image)
    save_image(image, f"metastatic-breast-cancer_preprocessed.jpg")
    blob_detection(image, sigma=5, threshold=0, filename="metastatic-breast-cancer")

