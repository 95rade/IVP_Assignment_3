import numpy as np
import math
import cv2
from Utilities import load_image, plot_image, plot_figures


def H(x, y, cx, cy, rad):
    f = (x - cx) ** 2 + (y - cy) ** 2
    f = math.exp(-(f / (2 * (rad ** 2))))
    return f


def high_pass_filter(img_name, filter_size):
    img = cv2.imread(img_name, 0)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    # fshift[crow - filter_size:crow + filter_size, ccol - filter_size:ccol + filter_size] = 0

    fshift_new = fshift.copy()

    for i in range(rows):
        for j in range(cols):
            fshift_new[i, j] = fshift[i, j] * (1 - H(i, j, crow, ccol, filter_size))

    f_ishift = np.fft.ifftshift(fshift + fshift_new)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back


def low_pass_filter(image_name, filter_size):
    img = cv2.imread(image_name, 0)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # create a mask first, center square is 1, remaining all zeros
    # mask = np.zeros((rows, cols), np.uint8)
    # mask[crow - filter_size:crow + filter_size, ccol - filter_size:ccol + filter_size] = 1
    # mask = np.array([[0 if sqrt((i - ccol)*(i - ccol) + (j - crow)*(j - crow)) <= filter_size else 1 for j in range(cols)] for i in range(rows) ])
    # fshift = fshift * mask

    for i in range(rows):
        for j in range(cols):
            fshift[i, j] = fshift[i, j] * (H(i, j, crow, ccol, filter_size))

    f_ishift = np.fft.ifftshift(fshift)
    d_shift = np.array(np.dstack([f_ishift.real, f_ishift.imag]))
    img_back = cv2.idft(d_shift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back


def main():
    img_high = high_pass_filter('images/97.jpg', 20)
    img_low = low_pass_filter('images/97.jpg', 16)

    figures = []

    figures.append(('Orignal image', load_image('images/97.jpg', type=0)))
    figures.append(('Low pass filter', img_low.copy()))
    figures.append(('High pass filter', img_high.copy()))

    plot_figures(figures, 2, 2)


if __name__ == "__main__":
    main()
