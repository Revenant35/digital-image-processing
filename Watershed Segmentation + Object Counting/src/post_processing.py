from skimage.morphology import opening, closing, disk


def post_process_mask(binary_mask, noise_removal_size=3, hole_filling_size=3):
    """
    This function applies morphological operations to clean up the mask.

    Parameters:
    - binary_mask: The binary mask obtained from k-means clustering.
    - noise_removal_size: The radius of the disk used for noise removal (opening). Default is 3.
    - hole_filling_size: The radius of the disk used for hole filling (closing). Default is 3.

    Returns:
    The cleaned binary mask after applying morphological operations.
    """

    # Create structuring elements for morphological operations
    noise_removal_element = disk(noise_removal_size)
    hole_filling_element = disk(hole_filling_size)

    # Remove noise (small objects) from the image
    # Opening is an erosion followed by a dilation
    opened_mask = opening(binary_mask, noise_removal_element)

    # Fill holes in the image
    # Closing is a dilation followed by an erosion
    closed_mask = closing(opened_mask, hole_filling_element)

    return closed_mask
