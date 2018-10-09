import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as read
import cv2

MIN_PIXEL_VALUE = 0
MAX_PIXEL_VALUE = 255


def load_image(name, usecv=True, type=None):
    if usecv:
        if type is not None:
            return cv2.imread(name, type)
        else:
            return cv2.imread(name)
    else:
        return read.imread(name)


def plot_image(image, usecv=True):
    if usecv:
        cv2.imshow('image', image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.show()


def plot_figures(figures, nrows=1, ncols=1):
    """Plot a dictionary of figures.
    Parameters
    ----------
    figures : list of (name, image)
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind, figure in enumerate(figures):
        name, image = figure
        axeslist.ravel()[ind].imshow(image, cmap=plt.gray())
        axeslist.ravel()[ind].set_title(name)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()  # optional
    plt.show()


def linear_transformation_to_pixel_value_range(data):
    max = np.max(data)
    min = np.min(data)
    data = np.array(data, dtype=np.float64)
    new_data = MAX_PIXEL_VALUE * ((data - min) / (max - min))
    return np.rint(new_data).astype(np.uint8)


def high_pass_filter(img_name, filter_size):
    img = cv2.imread(img_name, 0)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    fshift[crow - filter_size:crow + filter_size, ccol - filter_size:ccol + filter_size] = 0

    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    figures = []

    figures.append(('Input Image', img))
    figures.append(('High pass filter', img_back))

    plot_figures(figures, 1, 2)


def low_pass_filter(image_name, filter_size):
    img = cv2.imread(image_name, 0)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - filter_size:crow + filter_size, ccol - filter_size:ccol + filter_size] = 1

    # apply mask and inverse DFT
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    d_shift = np.array(np.dstack([f_ishift.real, f_ishift.imag]))
    img_back = cv2.idft(d_shift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    figures = []

    figures.append(('Input Image', img))
    figures.append(('High pass filter', img_back))

    plot_figures(figures, 1, 2)

if __name__ == "__main__":
    main()