import cv2
import numpy as np
from matplotlib import pyplot as plt

imageSource = 'images/img2.tif'
img = cv2.imread(imageSource, cv2.IMREAD_GRAYSCALE)
imageSource = 'images/img1original.tif'
img_orig = cv2.imread(imageSource, cv2.IMREAD_GRAYSCALE)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

f_orig = np.fft.fft2(img - img_orig)
fshift_orig = np.fft.fftshift(f_orig)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

fshift[crow, crow + 16] = np.average(fshift[crow -2: crow + 3, crow + 16 - 2: crow + 16 +3])
fshift[crow, crow - 16] = np.average(fshift[crow -2: crow + 3, crow - 16 - 2: crow - 16 +3])

# i = 16
# j = 1
#
# while i * j < fshift.shape[1] // 2:
#     fshift[crow, crow + (i * j)] = 0
#     fshift[crow, crow - (i * j)] = 0
#     j = j + 1


magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
magnitude_spectrum_orig = 20 * np.log(np.abs(fshift_orig) + 1)
f_ishift2 = np.fft.ifftshift(fshift)
mag_back2 = np.fft.ifft2(f_ishift2)
img_back2 = np.abs(mag_back2)

plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.imshow(magnitude_spectrum_orig, cmap='gray')
plt.title('Magnitude Spectrum of pattern'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3), plt.imshow(img_back2, cmap='gray')
plt.title('Output Image '), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])


plt.show()
