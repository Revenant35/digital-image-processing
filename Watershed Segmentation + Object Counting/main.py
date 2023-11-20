from pathlib import Path

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

import src.kmeans_clustering as kc
import src.post_processing as pp
import src.watershed_marker as wm

images_path = Path(__file__).parents[0] / 'images'
assert images_path.exists(), f'Path {images_path} does not exist.'


def main():

    image_path = images_path / 'GlomusTumor6_crp2.jpg'

    # Display the input image
    image = cv.imread(str(image_path))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title('Input image')
    plt.show()

    # TASK 1
    # Generate the binary mask
    binary_mask = kc.kmeans_image_segmentation(str(image_path), invert=True)

    # Display the mask
    plt.imshow(binary_mask, cmap='gray')
    plt.title('Binary mask')
    plt.show()

    # TASK 2
    # Post-process the mask
    unknown = pp.post_process_mask(binary_mask, 1, 2)

    # Display the post-processed mask
    plt.imshow(unknown, cmap='gray')
    plt.title('Post-processed mask')
    plt.show()

    # TASK 3
    # Find the markers for the watershed algorithm
    markers = wm.watershed_with_distance_transform(unknown)

    # Display the markers
    plt.imshow(markers.astype(np.uint8), cmap='gray')
    plt.title('Markers')
    plt.show()

    markers = cv.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]

    plt.imshow(image)
    plt.show()


    # for image_path in images_path.iterdir():
    #     # Display the input image
    #     image = plt.imread(str(image_path))
    #     plt.imshow(image)
    #     plt.show()
    #
    #     # Generate the binary mask
    #     binary_mask = kc.kmeans_image_segmentation(str(image_path))
    #
    #     # Display the mask
    #     plt.imshow(binary_mask, cmap='gray')
    #     plt.show()
    #
    #     # Post-process the mask
    #     post_processed_mask = pp.post_process_mask(binary_mask)
    #
    #     # Display the post-processed mask
    #     plt.imshow(post_processed_mask, cmap='gray')
    #     plt.show()
    #
    #     # Find the markers for the watershed algorithm
    #     markers = wm.watershed_with_distance_transform(post_processed_mask)
    #
    #     # Display the markers
    #     plt.imshow(markers, cmap='gray')
    #     plt.show()


if __name__ == '__main__':
    main()
