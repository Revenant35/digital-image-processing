import numpy as np
import cv2

# Mean Squared Error implementation
# MSE calculates the average squared difference between the corresponding pixels of two
# images. MSE provides a numerical value indicating similarity of two images and it is widely
# used to assess image quality. MSE between two images is defined as:

# MSE = 1/(M*N) * sum(sum((I1(i,j) - I2(i,j))^2))


def MSE(img1, img2):
    # Check if the images are of the same size
    if img1.shape != img2.shape:
        print("Images are not of the same size")
        return -1

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Calculate MSE
    M = img1.shape[0]
    N = img1.shape[1]

    error = 0

    for x in range(M):
        for y in range(N):
            error += (img1[x, y] - img2[x, y]) ** 2

    error /= M * N

    return error


# Test script to compare the two implementations
if __name__ == "__main__":
    image_1 = cv2.imread("../images/Test1.png", cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.imread("../images/Test1Noise1.png", cv2.IMREAD_GRAYSCALE)
    image_3 = cv2.imread("../images/Test1Noise2.png", cv2.IMREAD_GRAYSCALE)

    image_4 = cv2.imread("../images/Test2.png", cv2.IMREAD_GRAYSCALE)
    image_5 = cv2.imread("../images/Test2Noise2.png", cv2.IMREAD_GRAYSCALE)

    print("MSE of the image_1s is: ", MSE(image_1, image_1))
    print("MSE of image_1 and image_2 is: ", MSE(image_1, image_2))
    print("MSE of image_1 and image_3 is: ", MSE(image_1, image_3))
    print("MSE of image_2 and image_3 is: ", MSE(image_2, image_3))
    print("MSE of image_4 and image_5 is: ", MSE(image_4, image_5))
