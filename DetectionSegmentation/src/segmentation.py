import numpy as np
from file_operations import read_image, save_image
from cv2 import kmeans
import cv2


def segment_using_kmeans(image: np.ndarray[np.uint8], k: int = 2):
    """
    Segments an image using k-means clustering.

    Args:
    - image (numpy.ndarray): The 3-channel input image.
    - k (int): The number of clusters. Positive, non-zero integer.

    Returns:
    - numpy.ndarray: The segmented image.
    """

    assert image.shape[2] == 3, "Image must have 3 channels."
    assert k > 0, "k must be greater than 0."

    # reshape image into a 2D array of floating point intensity values
    data = np.float32(image.reshape((-1, 3)))

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 250, 0.90)

    # perform k-means clustering with random centers
    retval, labels, centers = kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    segmented_data = centers[labels.flatten()]

    # reshape data into the original image dimensions and data type
    segmented_image = segmented_data.reshape(image.shape)
    segmented_image = np.uint8(segmented_image)

    return segmented_image


if __name__ == "__main__":
    # Read image
    image = read_image("GlomusTumor6.jpg")

    # Segment Image
    image = segment_using_kmeans(image)
    save_image(image, "GlomusTumor6_segmented.jpg")

    # Threshold image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, image)
    save_image(image, "GlomusTumor6_segmented_mask.jpg")

    # Read image
    image = read_image("metastatic-breast-cancer.jpg")
    
    # Segment Image
    image = segment_using_kmeans(image)
    save_image(image, "metastatic-breast-cancer_segmented.jpg")

    # Threshold image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, image)
    save_image(image, "metastatic-breast-cancer_segmented_mask.jpg")
