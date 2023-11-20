import cv2 as cv
import numpy as np
from scipy import ndimage
from skimage import measure, filters, morphology
from skimage.feature import peak_local_max


def watershed_with_distance_transform(post_processed_mask):
    """
    This function uses a binary mask to determine internal and external markers using
    distance transform and then applies the watershed algorithm.

    Parameters:
    - binary_mask: The binary mask obtained from post-processing.

    Returns:
    - markers: An array the same size as `binary_mask` where each pixel has a label
               indicating the segment it belongs to.
    """

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(post_processed_mask, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding sure background area
    sure_bg = cv.dilate(post_processed_mask, np.ones((3, 3), np.uint8), iterations=3)
    sure_bg = cv.bitwise_not(sure_bg)

    # Finding unknown region
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    return markers
