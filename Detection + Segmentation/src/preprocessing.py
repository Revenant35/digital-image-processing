import cv2
from file_operations import read_image, show_image


def preprocess_image(image):
    return cv2.medianBlur(image, 5)


def gaussian_blur(image, kernel_size):  # I prefer kernel size = 5
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def median_blur(image, kernel_size):  # I prefer kernel size = 5
    return cv2.medianBlur(image, kernel_size)


if __name__ == "__main__":
    filenames = ["GlomusTumor6.jpg", "metastatic-breast-cancer.jpg"]
    for filename in filenames:
        image = read_image(filename)
        show_image(image, "Original Image")
        for i in range(3, 12, 2):
            blurred_image = gaussian_blur(image, i)
            show_image(blurred_image, f"Gaussian Blurred Image (kernel_size={i})")
        for i in range(3, 12, 2):
            blurred_image = median_blur(image, i)
            show_image(blurred_image, f"Median Blurred Image (kernel_size={i})")
