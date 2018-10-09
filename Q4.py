import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as read
import cv2
from Utilities import load_image, plot_image, \
    plot_figures, MAX_PIXEL_VALUE, MIN_PIXEL_VALUE, linear_transformation_to_pixel_value_range


def mean_filter(arr):
    return np.clip(np.rint(np.mean(arr)), MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)


def median_filter(arr):
    return np.rint(np.median(arr))


def max_filter(arr):
    return np.max(arr)


def min_filter(arr):
    return np.min(arr)


def apply_noise_removing_filter(img, filter_size, function):
    filter_size = filter_size // 2

    rows, cols = img.shape

    img_new = img.copy()

    for r in range(filter_size, rows - filter_size):
        for c in range(filter_size, cols - filter_size):
            img_new[r, c] = function(img[r - filter_size: r + filter_size + 1, c - filter_size: c + filter_size + 1])

    return img_new.astype(np.uint8)


def mean_square_error(img1, img2):
    if img1.shape != img2.shape:
        print("Cannot calculate mean square error as both images does not have same shape")
        return

    rows, cols = img2.shape
    error = 0

    err_img = np.zeros((rows, cols))

    for r in range(rows):
        for c in range(cols):
            err_img[r, c] = (img1[r, c].astype(np.float64) - img2[r, c].astype(np.float64)) ** 2
            error += err_img[r, c]

    error /= rows * cols

    max_err = np.max(err_img)
    min_err = np.min(err_img)

    err_img = ((err_img - min_err) / (max_err - min_err)) * MAX_PIXEL_VALUE

    return err_img.astype(np.uint8), error


def remove_noise(img, img_name, filter_size, filter_function):
    img_filtered = apply_noise_removing_filter(img.copy(), filter_size, filter_function)

    img_err, err_left_mean = mean_square_error(img.copy(), img_filtered.copy())

    img_diff = img_filtered.astype(np.int) - img.astype(np.int)
    img_diff = linear_transformation_to_pixel_value_range(img_diff)

    figures = []

    figures.append((img_name + ' orginal image', img))
    figures.append((str(filter_function)[9:-19] + 'ed', img_filtered))
    figures.append(('Error image', img_diff))

    print('Mean square error of ' + img_name + ' image after applying ' + str(filter_function)[9:-19] + ' = ' + str(err_left_mean))

    plot_figures(figures, 1, 3)


def main():
    img = load_image('images/img1noisy.tif', type=0)

    rows, cols = img.shape
    rows = rows // 2
    cols = cols // 2

    img_left = img[:, : cols + 1]
    img_right = img[:, cols:]

    remove_noise(img_left, 'Left', 3, max_filter)
    remove_noise(img_left, 'Left', 3, min_filter)
    remove_noise(img_left, 'Left', 3, mean_filter)
    remove_noise(img_left, 'Left', 3, median_filter)

    remove_noise(img_right, 'Right', 3, max_filter)
    remove_noise(img_right, 'Right', 3, min_filter)
    remove_noise(img_right, 'Right', 3, mean_filter)
    remove_noise(img_right, 'Right', 3, median_filter)


if __name__ == "__main__":
    main()
