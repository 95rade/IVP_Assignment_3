import sys
import cv2
import numpy as np
from Utilities import plot_image, load_image, plot_figures


def add_gaussian_noise(img, mean, variance):
    sigma = variance ** 0.5
    gauss = np.ndarray(img.shape, dtype=np.uint8)
    cv2.randn(gauss, mean, sigma)
    noisy = (img + gauss).astype(np.uint8)

    return noisy


def add_salt_and_pepper_noise(img, amount, salt_prob):
    img = np.copy(img)
    num_of_pixels_for_salt = np.ceil(amount * img.size * salt_prob)

    for i in range(int(num_of_pixels_for_salt)):
        x = np.random.randint(0, img.shape[0] - 1)
        y = np.random.randint(0, img.shape[1] - 1)
        img[x, y] = 255

    num_of_pixels_for_pepper = np.ceil(amount * img.size * (1. - salt_prob))

    for i in range(int(num_of_pixels_for_pepper)):
        x = np.random.randint(0, img.shape[0] - 1)
        y = np.random.randint(0, img.shape[1] - 1)
        img[x, y] = 0

    return img


def main():
    imageSource = 'images/img1original.tif'
    image = load_image(imageSource, type=0)

    gaussian_noise_image = add_gaussian_noise(image.copy(), 0, 300)
    salt_and_peper_noise_image = add_salt_and_pepper_noise(image.copy(), 0.04, 0.5)

    figures = []

    figures.append(('Orignal Image', image.copy()))
    figures.append(('Gaussian noise Image', gaussian_noise_image.copy()))
    figures.append(('Salt and Pepper Image', salt_and_peper_noise_image.copy()))

    plot_figures(figures, 1, 3)


if __name__ == "__main__":
    main()
