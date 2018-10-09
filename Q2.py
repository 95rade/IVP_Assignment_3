from Utilities import load_image, plot_image
import numpy as np
from Utilities import linear_transformation_to_pixel_value_range


def main():
    # sobel in x direction
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    # sobel in y direction
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    img = load_image('images/two_cats.jpg', type=0)

    # img_dft = compute_dft(img)
    img_dft = np.fft.fft2(img)
    sobelx_filter_dft = np.fft.fft2(sobel_x, s=img.shape)
    sobely_filter_dft = np.fft.fft2(sobel_y, s=img.shape)
    filteredx_img_dft = img_dft * sobelx_filter_dft
    filteredy_img_dft = img_dft * sobely_filter_dft

    filteredx_img_back = np.fft.ifft2(filteredx_img_dft)
    filteredy_img_back = np.fft.ifft2(filteredy_img_dft)
    ans = filteredy_img_back.real + filteredx_img_back.real

    ans = linear_transformation_to_pixel_value_range(ans)

    plot_image(ans, False)


if __name__ == "__main__":
    main()
