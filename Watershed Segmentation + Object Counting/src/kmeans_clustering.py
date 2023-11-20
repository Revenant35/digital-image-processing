import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def kmeans_image_segmentation(image_path, n_clusters=2, invert=False):
    """
    This function applies K-means clustering to an image for segmentation.

    Parameters:
    - image_path: The path to the image file.
    - n_clusters: The number of clusters to form. Default is 2 for binary mask.

    Returns:
    A binary mask of the image where the foreground is 1 and the background is 0.
    """

    image = cv2.imread(image_path)
    # Convert from BGR to RGB (OpenCV uses BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # Convert to float
    pixel_values = np.float32(pixel_values)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(pixel_values, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8 bit values
    centers = np.uint8(centers)

    # Map label to pixel values
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image dimensions
    segmented_image = segmented_image.reshape(image.shape)

    # Determine which cluster corresponds to the background
    # We assume the background is the most frequent color
    counts = np.bincount(labels.flatten())
    background_label = np.argmax(counts)

    # Create a mask where the background cluster is 0 and the rest is 1
    if invert:
        mask = np.where(labels.flatten() == background_label, 1, 0)
    else:
        mask = np.where(labels.flatten() == background_label, 0, 1)

    # Reshape the mask back to the original image shape
    mask = mask.reshape(image.shape[:2])

    return mask.astype(np.uint8)
